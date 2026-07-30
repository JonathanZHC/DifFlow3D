"""DifFlow3D benchmark with dense first-level motion recovery.

Pipeline for each incoming frame:
    synthetic/raw CPU cloud (configurable, e.g. 300k points)
      -> pinned-host staging and H2D copy
      -> mandatory fixed-resolution GPU voxel downsampling
      -> optional adaptive second GPU voxel downsampling
      -> exact GPU farthest-point sampling (FPS)
      -> existing DifFlow3D streaming CUDA-Graph runner
      -> all-anchor inverse-distance or Gaussian-softmax recovery
         of flow and velocity for every first-level voxel representative

The script reports component and end-to-end timing, anchor-level accuracy,
recovered first-level accuracy, deadline misses, and GPU memory. Synthetic
frames are generated online and outside the measured perception cycle. RViz2
publishing is also outside all benchmark timers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation
import torch


# Before using, the pcds should be scaled to 1*1*1 m^3 match the real-world dimensions of the scene.
DIMENSION_FACTOR = 1.0
MOTION = True


MotionMode = Literal["static", "sinusoidal", "twist"]

# Same default as the supplied scene generator:
# 0.017 m is interpreted as three standard deviations.
DEFAULT_SENSOR_NOISE_STD_M = 0.017 / 3.0 * DIMENSION_FACTOR 



@dataclass(frozen=True)
class SQSpec:
    name: str
    axes: tuple[float, float, float]
    eps1: float
    eps2: float
    initial_translation: tuple[float, float, float]
    initial_axis_angle: tuple[float, float, float]
    points_per_frame: int
    motion_mode: MotionMode
    motion_amplitude: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    motion_frequency: float = 0.0
    motion_phase: float = 0.0
    twist_world: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


OBSTACLE_SPECS = (
    SQSpec(
        name="main_box",
        axes=(0.18 * DIMENSION_FACTOR , 0.55 * DIMENSION_FACTOR, 0.40 * DIMENSION_FACTOR),
        eps1=1.0,
        eps2=1.0,
        initial_translation=(0.25 * DIMENSION_FACTOR, 0.0, 0.30 * DIMENSION_FACTOR),
        initial_axis_angle=(0.0, 0.0, 0.0),
        points_per_frame=1024,
        motion_mode="static",
    ),
    SQSpec(
        name="left_rounded_pillar",
        axes=(0.12 * DIMENSION_FACTOR, 0.12 * DIMENSION_FACTOR, 0.55 * DIMENSION_FACTOR),
        eps1=0.45,
        eps2=1.0,
        initial_translation=(0.15 * DIMENSION_FACTOR, 0.42 * DIMENSION_FACTOR, 0.35 * DIMENSION_FACTOR),
        initial_axis_angle=(0.0, 0.35, 0.15),
        points_per_frame=1024,
        motion_mode="sinusoidal" if MOTION else "static",
        motion_amplitude=(0.0, 5 * 0.20 * DIMENSION_FACTOR, 0.0, 0.0, 0.0, 0.0),
        motion_frequency=0.15,
        motion_phase=0.0,
    ),
    SQSpec(
        name="right_flat_ellipsoid",
        axes=(0.35 * DIMENSION_FACTOR, 0.16 * DIMENSION_FACTOR, 0.14 * DIMENSION_FACTOR),
        eps1=1.0,
        eps2=1.0,
        initial_translation=(0.45 * DIMENSION_FACTOR, -0.38 * DIMENSION_FACTOR, 0.26 * DIMENSION_FACTOR),
        initial_axis_angle=(0.25, 0.0, -0.55),
        points_per_frame=1024,
        motion_mode="sinusoidal" if MOTION else "static",
        motion_amplitude=(0.0, 0.0, 0.0, 0.0, 0.0, 5 * 0.50),
        motion_frequency=0.20,
        motion_phase=0.0,
    ),
    SQSpec(
        name="front_tilted_box",
        axes=(0.22 * DIMENSION_FACTOR, 0.18 * DIMENSION_FACTOR, 0.28 * DIMENSION_FACTOR),
        eps1=0.35,
        eps2=0.35,
        initial_translation=(0.62 * DIMENSION_FACTOR, 0.12 * DIMENSION_FACTOR, 0.42 * DIMENSION_FACTOR),
        initial_axis_angle=(0.45, -0.25, 0.75),
        points_per_frame=1024,
        motion_mode="twist" if MOTION else "static",
        twist_world=(10 * 0.02 * DIMENSION_FACTOR, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
)


@dataclass(frozen=True)
class SceneSequence:
    points: np.ndarray
    timestamps_s: np.ndarray

    # Exact displacement of each source material point:
    # p(t + dt) - p(t)
    gt_flow: np.ndarray

    # Analytic instantaneous velocity of the same material point
    # evaluated at the target time t + dt.
    gt_velocity_target: np.ndarray

    object_ids: np.ndarray
    object_names: tuple[str, ...]
    dynamic_mask: np.ndarray
    max_gt_flow_by_object_m: dict[str, float]


def signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * np.abs(values) ** exponent


def make_superquadric_mesh(
    spec: SQSpec,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    if resolution < 8:
        raise ValueError("--mesh-resolution must be at least 8.")

    n_eta = resolution + 1
    n_omega = 2 * resolution

    eta = np.linspace(
        -0.5 * np.pi,
        0.5 * np.pi,
        n_eta,
        dtype=np.float64,
    )
    omega = np.linspace(
        -np.pi,
        np.pi,
        n_omega,
        endpoint=False,
        dtype=np.float64,
    )
    eta_grid, omega_grid = np.meshgrid(eta, omega, indexing="ij")

    cos_eta = signed_power(np.cos(eta_grid), spec.eps1)
    sin_eta = signed_power(np.sin(eta_grid), spec.eps1)
    cos_omega = signed_power(np.cos(omega_grid), spec.eps2)
    sin_omega = signed_power(np.sin(omega_grid), spec.eps2)

    a1, a2, a3 = spec.axes
    vertices = np.stack(
        (
            a1 * cos_eta * cos_omega,
            a2 * cos_eta * sin_omega,
            a3 * sin_eta,
        ),
        axis=-1,
    ).reshape(-1, 3)

    faces: list[tuple[int, int, int]] = []
    for eta_index in range(n_eta - 1):
        row0 = eta_index * n_omega
        row1 = (eta_index + 1) * n_omega
        for omega_index in range(n_omega):
            omega_next = (omega_index + 1) % n_omega
            v00 = row0 + omega_index
            v01 = row0 + omega_next
            v10 = row1 + omega_index
            v11 = row1 + omega_next
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    return (
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int32),
    )


def prepare_triangle_sampler(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    double_area = np.linalg.norm(np.cross(edge1, edge2), axis=1)

    valid = np.isfinite(double_area) & (double_area > 1.0e-15)
    if not np.any(valid):
        raise ValueError("Superquadric mesh has no non-degenerate triangles.")

    triangles = triangles[valid]
    probabilities = double_area[valid]
    probabilities = probabilities / probabilities.sum()

    return (
        triangles[:, 0],
        triangles[:, 1],
        triangles[:, 2],
        probabilities,
    )


def sample_points_on_triangle_mesh(
    sampler: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    v0, v1, v2, probabilities = sampler

    triangle_indices = rng.choice(
        len(probabilities),
        size=count,
        replace=True,
        p=probabilities,
    )
    u = rng.random(count)
    v = rng.random(count)
    sqrt_u = np.sqrt(u)

    weights0 = 1.0 - sqrt_u
    weights1 = sqrt_u * (1.0 - v)
    weights2 = sqrt_u * v

    points = (
        weights0[:, None] * v0[triangle_indices]
        + weights1[:, None] * v1[triangle_indices]
        + weights2[:, None] * v2[triangle_indices]
    )
    return np.ascontiguousarray(points, dtype=np.float64)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def so3_left_jacobian(rotvec: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian J_l(phi).

    For R(t) = Exp(phi(t)^), the world-frame angular velocity is

        omega_world = J_l(phi) @ phi_dot.
    """

    phi = np.asarray(rotvec, dtype=np.float64)
    theta_squared = float(phi @ phi)
    phi_hat = skew(phi)

    if theta_squared < 1.0e-12:
        return (
            np.eye(3, dtype=np.float64)
            + 0.5 * phi_hat
            + (1.0 / 6.0) * (phi_hat @ phi_hat)
        )

    theta = np.sqrt(theta_squared)

    coefficient_1 = (
        1.0 - np.cos(theta)
    ) / theta_squared

    coefficient_2 = (
        theta - np.sin(theta)
    ) / (theta_squared * theta)

    return (
        np.eye(3, dtype=np.float64)
        + coefficient_1 * phi_hat
        + coefficient_2 * (phi_hat @ phi_hat)
    )


def motion_state_at_time(
    spec: SQSpec,
    time_s: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return translation, rotation, linear velocity and angular velocity.

    Both velocities are expressed in the world frame.
    """

    translation0 = np.asarray(
        spec.initial_translation,
        dtype=np.float64,
    )
    rotvec0 = np.asarray(
        spec.initial_axis_angle,
        dtype=np.float64,
    )

    if spec.motion_mode == "static":
        rotation = Rotation.from_rotvec(
            rotvec0
        ).as_matrix()

        return (
            translation0.copy(),
            rotation,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )

    if spec.motion_mode == "sinusoidal":
        amplitude = np.asarray(
            spec.motion_amplitude,
            dtype=np.float64,
        )

        angular_frequency = (
            2.0 * np.pi * spec.motion_frequency
        )

        phase = (
            angular_frequency * time_s
            + spec.motion_phase
        )

        pose_offset = amplitude * np.sin(phase)

        # Analytic time derivative of pose_offset.
        pose_rate = (
            amplitude
            * angular_frequency
            * np.cos(phase)
        )

        translation = (
            translation0 + pose_offset[:3]
        )
        linear_velocity_world = pose_rate[:3]

        rotvec = rotvec0 + pose_offset[3:]
        rotvec_rate = pose_rate[3:]

        rotation = Rotation.from_rotvec(
            rotvec
        ).as_matrix()

        angular_velocity_world = (
            so3_left_jacobian(rotvec)
            @ rotvec_rate
        )

        return (
            translation,
            rotation,
            linear_velocity_world,
            angular_velocity_world,
        )

    if spec.motion_mode == "twist":
        twist = np.asarray(
            spec.twist_world,
            dtype=np.float64,
        )

        linear_velocity_world = twist[:3]
        angular_velocity_world = twist[3:]

        translation = (
            translation0
            + linear_velocity_world * time_s
        )

        rotation0 = Rotation.from_rotvec(rotvec0)

        # Left multiplication means angular_velocity_world is a
        # world/spatial angular velocity.
        delta_rotation_world = Rotation.from_rotvec(
            angular_velocity_world * time_s
        )

        rotation = (
            delta_rotation_world * rotation0
        ).as_matrix()

        return (
            translation,
            rotation,
            linear_velocity_world,
            angular_velocity_world,
        )

    raise ValueError(
        f"Unsupported motion mode: {spec.motion_mode}"
    )


def analytic_point_velocity_world(
    local_points: np.ndarray,
    rotation_world_object: np.ndarray,
    linear_velocity_world: np.ndarray,
    angular_velocity_world: np.ndarray,
) -> np.ndarray:
    """Analytic world-frame velocity of rigid-body surface points."""

    # Relative position from object center, expressed in world frame.
    lever_arm_world = (
        local_points @ rotation_world_object.T
    )

    angular_velocity = np.broadcast_to(
        angular_velocity_world,
        lever_arm_world.shape,
    )

    rotational_velocity = np.cross(
        angular_velocity,
        lever_arm_world,
    )

    point_velocity_world = (
        linear_velocity_world[None, :]
        + rotational_velocity
    )

    return np.ascontiguousarray(
        point_velocity_world,
        dtype=np.float64,
    )


def pose_at_time(
    spec: SQSpec,
    time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    translation0 = np.asarray(spec.initial_translation, dtype=np.float64)
    rotvec0 = np.asarray(spec.initial_axis_angle, dtype=np.float64)

    if spec.motion_mode == "static":
        return translation0.copy(), Rotation.from_rotvec(rotvec0).as_matrix()

    if spec.motion_mode == "sinusoidal":
        amplitude = np.asarray(spec.motion_amplitude, dtype=np.float64)
        phase = (
            2.0 * np.pi * spec.motion_frequency * time_s
            + spec.motion_phase
        )
        pose_offset = amplitude * np.sin(phase)
        translation = translation0 + pose_offset[:3]
        rotvec = rotvec0 + pose_offset[3:]
        return translation, Rotation.from_rotvec(rotvec).as_matrix()

    if spec.motion_mode == "twist":
        twist = np.asarray(spec.twist_world, dtype=np.float64)
        linear_velocity = twist[:3]
        angular_velocity = twist[3:]

        translation = translation0 + linear_velocity * time_s
        rotation0 = Rotation.from_rotvec(rotvec0)
        delta_rotation_world = Rotation.from_rotvec(
            angular_velocity * time_s
        )
        rotation = delta_rotation_world * rotation0
        return translation, rotation.as_matrix()

    raise ValueError(f"Unsupported motion mode: {spec.motion_mode}")


def transform_points(
    points_local: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    return points_local @ rotation.T + translation
def distribute_point_counts(total_points: int) -> list[int]:
    """Distribute an exact total across objects using their configured weights."""
    if total_points < len(OBSTACLE_SPECS):
        raise ValueError(
            f"--all-points must be at least {len(OBSTACLE_SPECS)}."
        )
    weights = np.asarray(
        [spec.points_per_frame for spec in OBSTACLE_SPECS],
        dtype=np.float64,
    )
    exact = weights / weights.sum() * float(total_points)
    counts = np.floor(exact).astype(np.int64)
    remainder = int(total_points - int(counts.sum()))
    if remainder:
        order = np.argsort(-(exact - counts))
        counts[order[:remainder]] += 1
    return [int(value) for value in counts]


def generate_sequence(
    *,
    frame_count: int,
    dt_s: float,
    mesh_resolution: int,
    total_points: int,
    sensor_noise_std_m: float,
    same_samples_across_frames: bool,
    seed: int,
) -> SceneSequence:
    if frame_count < 3:
        raise ValueError("--frames must be at least 3.")
    if dt_s <= 0.0:
        raise ValueError("Frame interval must be positive.")
    if sensor_noise_std_m < 0.0:
        raise ValueError("--sensor-noise-std must be non-negative.")

    rng = np.random.default_rng(seed)
    timestamps_s = np.arange(frame_count, dtype=np.float64) * dt_s
    counts = distribute_point_counts(total_points)

    samplers: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    fixed_samples: dict[str, np.ndarray] = {}

    for spec, count in zip(OBSTACLE_SPECS, counts):
        vertices, faces = make_superquadric_mesh(spec, mesh_resolution)
        samplers[spec.name] = prepare_triangle_sampler(vertices, faces)
        if same_samples_across_frames:
            fixed_samples[spec.name] = sample_points_on_triangle_mesh(
                samplers[spec.name], count, rng
            )

    points = np.empty((frame_count, total_points, 3), dtype=np.float32)
    gt_flow = np.empty((frame_count - 1, total_points, 3), dtype=np.float32)
    gt_velocity_target = np.empty(
        (frame_count - 1, total_points, 3), dtype=np.float32
    )

    object_ids = np.concatenate(
        [
            np.full(count, object_index, dtype=np.int16)
            for object_index, count in enumerate(counts)
        ]
    )
    dynamic_mask = np.concatenate(
        [
            np.full(count, spec.motion_mode != "static", dtype=bool)
            for spec, count in zip(OBSTACLE_SPECS, counts)
        ]
    )
    max_gt_flow_by_object_m = {
        spec.name: 0.0 for spec in OBSTACLE_SPECS
    }

    for frame_index, time_s in enumerate(timestamps_s):
        point_offset = 0
        for spec, count in zip(OBSTACLE_SPECS, counts):
            local_points = (
                fixed_samples[spec.name]
                if same_samples_across_frames
                else sample_points_on_triangle_mesh(
                    samplers[spec.name], count, rng
                )
            )

            (
                translation,
                rotation,
                _linear_velocity_world,
                _angular_velocity_world,
            ) = motion_state_at_time(spec, float(time_s))

            clean_points = transform_points(
                local_points, translation, rotation
            )
            observed_points = clean_points + rng.normal(
                loc=0.0,
                scale=sensor_noise_std_m,
                size=clean_points.shape,
            )

            point_slice = slice(point_offset, point_offset + count)
            points[frame_index, point_slice] = observed_points.astype(
                np.float32
            )

            if frame_index < frame_count - 1:
                target_time_s = float(time_s + dt_s)
                (
                    next_translation,
                    next_rotation,
                    next_linear_velocity_world,
                    next_angular_velocity_world,
                ) = motion_state_at_time(spec, target_time_s)

                corresponding_next_points = transform_points(
                    local_points, next_translation, next_rotation
                )
                object_flow = corresponding_next_points - clean_points
                gt_flow[frame_index, point_slice] = object_flow.astype(
                    np.float32
                )
                max_gt_flow_by_object_m[spec.name] = max(
                    max_gt_flow_by_object_m[spec.name],
                    float(
                        np.linalg.norm(object_flow, axis=1).max(initial=0.0)
                    ),
                )

                object_velocity_target = analytic_point_velocity_world(
                    local_points=local_points,
                    rotation_world_object=next_rotation,
                    linear_velocity_world=next_linear_velocity_world,
                    angular_velocity_world=next_angular_velocity_world,
                )
                gt_velocity_target[
                    frame_index, point_slice
                ] = object_velocity_target.astype(np.float32)

            point_offset += count

    return SceneSequence(
        points=np.ascontiguousarray(points),
        timestamps_s=timestamps_s,
        gt_flow=np.ascontiguousarray(gt_flow),
        gt_velocity_target=np.ascontiguousarray(gt_velocity_target),
        object_ids=object_ids,
        object_names=tuple(spec.name for spec in OBSTACLE_SPECS),
        dynamic_mask=dynamic_mask,
        max_gt_flow_by_object_m=max_gt_flow_by_object_m,
    )


def cuda_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device}.")
    return (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(cuda_index(device))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def safe_mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(values.mean())


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": percentile(array, 90.0),
        "p95": percentile(array, 95.0),
        "p99": percentile(array, 99.0),
        "max": float(array.max()),
    }


def print_timing_row(name: str, stats: dict[str, float]) -> None:
    if not stats:
        return
    print(
        f"{name:24s} "
        f"mean={stats['mean']:8.3f} ms  "
        f"med={stats['median']:8.3f}  "
        f"p95={stats['p95']:8.3f}  "
        f"max={stats['max']:8.3f}"
    )



@dataclass(frozen=True)
class OnlineSceneFrame:
    """One synthetic sensor frame and source-point GT for the next frame."""

    points: np.ndarray
    timestamp_s: float
    gt_flow_to_next: np.ndarray
    gt_velocity_target: np.ndarray
    object_ids: np.ndarray
    dynamic_mask: np.ndarray


class OnlineSceneGenerator:
    """Generate one scene frame at a time instead of precomputing F x N arrays.

    Synthetic frame generation is treated as sensor acquisition and is kept
    outside the measured perception cycle.  A deterministic per-frame RNG makes
    results reproducible even when warmup frames are requested out of order.
    """

    def __init__(
        self,
        *,
        frame_count: int,
        dt_s: float,
        mesh_resolution: int,
        total_points: int,
        sensor_noise_std_m: float,
        same_samples_across_frames: bool,
        seed: int,
    ) -> None:
        if frame_count < 3:
            raise ValueError("--frames must be at least 3.")
        if dt_s <= 0.0:
            raise ValueError("Frame interval must be positive.")
        if sensor_noise_std_m < 0.0:
            raise ValueError("--sensor-noise-std must be non-negative.")

        self.frame_count = int(frame_count)
        self.dt_s = float(dt_s)
        self.total_points = int(total_points)
        self.sensor_noise_std_m = float(sensor_noise_std_m)
        self.same_samples_across_frames = bool(same_samples_across_frames)
        self.seed = int(seed)
        self.counts = distribute_point_counts(total_points)

        self.samplers: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self.fixed_samples: dict[str, np.ndarray] = {}

        base_rng = np.random.default_rng(self.seed)
        for spec, count in zip(OBSTACLE_SPECS, self.counts):
            vertices, faces = make_superquadric_mesh(spec, mesh_resolution)
            sampler = prepare_triangle_sampler(vertices, faces)
            self.samplers[spec.name] = sampler
            if self.same_samples_across_frames:
                self.fixed_samples[spec.name] = sample_points_on_triangle_mesh(
                    sampler,
                    count,
                    base_rng,
                )

        self.object_ids = np.concatenate(
            [
                np.full(count, object_index, dtype=np.int16)
                for object_index, count in enumerate(self.counts)
            ]
        )
        self.dynamic_mask = np.concatenate(
            [
                np.full(count, spec.motion_mode != "static", dtype=bool)
                for spec, count in zip(OBSTACLE_SPECS, self.counts)
            ]
        )

    def frame(self, frame_index: int) -> OnlineSceneFrame:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(
                f"Frame index {frame_index} is outside [0,{self.frame_count})."
            )

        # Independent deterministic stream for each frame.
        rng = np.random.default_rng(self.seed + 104729 * frame_index + 17)
        time_s = float(frame_index) * self.dt_s

        points = np.empty((self.total_points, 3), dtype=np.float32)
        gt_flow = np.empty((self.total_points, 3), dtype=np.float32)
        gt_velocity_target = np.empty(
            (self.total_points, 3), dtype=np.float32
        )

        point_offset = 0
        for spec, count in zip(OBSTACLE_SPECS, self.counts):
            local_points = (
                self.fixed_samples[spec.name]
                if self.same_samples_across_frames
                else sample_points_on_triangle_mesh(
                    self.samplers[spec.name],
                    count,
                    rng,
                )
            )

            translation, rotation, _, _ = motion_state_at_time(spec, time_s)
            clean_points = transform_points(
                local_points,
                translation,
                rotation,
            )
            observed_points = clean_points + rng.normal(
                loc=0.0,
                scale=self.sensor_noise_std_m,
                size=clean_points.shape,
            )

            target_time_s = time_s + self.dt_s
            (
                next_translation,
                next_rotation,
                next_linear_velocity_world,
                next_angular_velocity_world,
            ) = motion_state_at_time(spec, target_time_s)

            corresponding_next_points = transform_points(
                local_points,
                next_translation,
                next_rotation,
            )
            object_flow = corresponding_next_points - clean_points
            object_velocity_target = analytic_point_velocity_world(
                local_points=local_points,
                rotation_world_object=next_rotation,
                linear_velocity_world=next_linear_velocity_world,
                angular_velocity_world=next_angular_velocity_world,
            )

            point_slice = slice(point_offset, point_offset + count)
            points[point_slice] = observed_points.astype(np.float32)
            gt_flow[point_slice] = object_flow.astype(np.float32)
            gt_velocity_target[point_slice] = object_velocity_target.astype(
                np.float32
            )
            point_offset += count

        return OnlineSceneFrame(
            points=np.ascontiguousarray(points),
            timestamp_s=time_s,
            gt_flow_to_next=np.ascontiguousarray(gt_flow),
            gt_velocity_target=np.ascontiguousarray(gt_velocity_target),
            object_ids=self.object_ids,
            dynamic_mask=self.dynamic_mask,
        )


@dataclass(frozen=True)
class PreparedPointCloudFrame:
    raw_points_cpu: np.ndarray
    first_downsample_points: torch.Tensor
    first_raw_indices: torch.Tensor
    candidate_points: torch.Tensor
    candidate_raw_indices: torch.Tensor
    anchor_points: torch.Tensor
    anchor_raw_indices: torch.Tensor
    timestamp_s: float
    raw_count: int
    first_count: int
    candidate_count: int
    unique_anchor_count: int
    second_downsample_enabled: bool
    first_voxel_resolution_m: float
    second_voxel_resolution_m: float | None
    host_stage_ms: float
    h2d_ms: float
    first_downsample_ms: float
    second_downsample_ms: float
    fps_ms: float
    preprocess_gpu_ms: float
    preprocess_wall_ms: float


@dataclass(frozen=True)
class DifFlow3DConfig:
    repo_path: Path
    checkpoint_path: Path
    model_module: str = "model_difflow"
    enable_tf32: bool = True
    cuda_graph_warmup: int = 10
    device: str = "cuda:0"
    num_points: int = 2048
    iters: int = 4
    uncertainty: float = 0.2
    strict_checkpoint: bool = True
    disable_bn_running_stats: bool = True
    frame_dt_s: float = 1.0 / 30.0
    max_frame_gap_s: float | None = None


@dataclass(frozen=True)
class CheckpointReport:
    is_exact: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


@dataclass(frozen=True)
class DifFlow3DEstimate:
    source_points: torch.Tensor
    warped_points: torch.Tensor
    residual_flow: torch.Tensor
    velocity: torch.Tensor
    valid_indices: torch.Tensor
    source_timestamp_s: float
    target_timestamp_s: float


@dataclass(frozen=True)
class DenseMotionRecovery:
    flow: torch.Tensor
    velocity: torch.Tensor


class GpuVoxelDownsampler:
    """Exact GPU voxel grouping retaining one real input point per cell."""

    def __init__(self, voxel_size_m: float) -> None:
        if voxel_size_m <= 0.0:
            raise ValueError("Voxel resolution must be positive.")
        self.voxel_size_m = float(voxel_size_m)

    def __call__(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected CUDA points [N,3], got {tuple(points.shape)}."
            )
        if points.shape[0] == 0:
            raise ValueError("Cannot voxelize an empty point cloud.")

        coordinates = torch.floor(
            points / self.voxel_size_m
        ).to(torch.int64)
        shifted = coordinates - coordinates.amin(dim=0)
        extents = shifted.amax(dim=0) + 1

        keys = (
            shifted[:, 0] * (extents[1] * extents[2])
            + shifted[:, 1] * extents[2]
            + shifted[:, 2]
        )
        sorted_keys, order = torch.sort(keys)
        keep = torch.ones_like(sorted_keys, dtype=torch.bool)
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
        representative_indices = order[keep]
        retained = points.index_select(0, representative_indices)
        return retained.contiguous(), representative_indices.contiguous()


class GpuFarthestPointSampler:
    def __init__(
        self,
        *,
        repo_path: Path,
        num_points: int,
        shortfall_policy: str,
    ) -> None:
        if num_points < 1:
            raise ValueError("--fps-points must be positive.")
        if shortfall_policy not in {"error", "repeat"}:
            raise ValueError("Unsupported FPS shortfall policy.")
        self.num_points = int(num_points)
        self.shortfall_policy = shortfall_policy

        repo_string = str(repo_path.expanduser().resolve())
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        importlib.invalidate_caches()

        errors: list[str] = []
        module = None
        for module_name in (
            "pointnet2.pointnet2_utils",
            "pointnet2_utils",
        ):
            try:
                module = importlib.import_module(module_name)
                break
            except Exception as error:
                errors.append(f"{module_name}: {error}")
        if module is None:
            raise RuntimeError(
                "Could not import the DifFlow3D PointNet++ FPS extension. "
                "Build/install pointnet2 first. Attempts: "
                + " | ".join(errors)
            )

        self._fps = getattr(module, "furthest_point_sample", None)
        if self._fps is None:
            raise RuntimeError(
                "PointNet++ module has no furthest_point_sample function."
            )

    def __call__(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        count = int(points.shape[0])
        if count < 1:
            raise ValueError("Cannot run FPS on an empty point cloud.")

        if count < self.num_points:
            if self.shortfall_policy == "error":
                raise RuntimeError(
                    f"FPS received {count} candidates, fewer than "
                    f"--fps-points={self.num_points}. Reduce the second "
                    "voxel resolution, disable the second downsampling, "
                    "reduce FPS count, or use --fps-shortfall repeat."
                )
            base = torch.arange(
                count,
                device=points.device,
                dtype=torch.long,
            )
            repeats = (self.num_points + count - 1) // count
            indices = base.repeat(repeats)[: self.num_points]
            return (
                points.index_select(0, indices).contiguous(),
                indices,
                count,
            )

        batched = points.unsqueeze(0).contiguous()
        indices = self._fps(batched, self.num_points)[0].long()
        gathered = points.index_select(0, indices)
        return gathered.contiguous(), indices.contiguous(), self.num_points


class FramePreprocessor:
    """Mandatory voxel-1 -> optional adaptive voxel-2 -> FPS pipeline."""

    def __init__(
        self,
        *,
        raw_point_count: int,
        device: torch.device,
        first_voxel_size_m: float,
        enable_second_downsample: bool,
        second_voxel_size_m: float | None,
        second_candidate_ratio: float,
        second_auto_dimension: float,
        second_auto_iterations: int,
        second_auto_tolerance: float,
        fps_sampler: GpuFarthestPointSampler,
    ) -> None:
        if raw_point_count < 1:
            raise ValueError("raw_point_count must be positive.")
        if first_voxel_size_m <= 0.0:
            raise ValueError("--voxel-resolution must be positive.")
        if second_voxel_size_m is not None and second_voxel_size_m <= 0.0:
            raise ValueError("--second-voxel-resolution must be positive.")
        if second_candidate_ratio <= 1.0:
            raise ValueError("--second-candidate-ratio must be > 1.")
        if second_auto_dimension <= 0.0:
            raise ValueError("--second-auto-dimension must be positive.")
        if second_auto_iterations < 1:
            raise ValueError("--second-auto-iterations must be positive.")
        if not 0.0 < second_auto_tolerance < 1.0:
            raise ValueError(
                "--second-auto-tolerance must lie strictly between 0 and 1."
            )

        self.device = device
        self.raw_point_count = int(raw_point_count)
        self.enable_second_downsample = bool(enable_second_downsample)
        self.first_voxel_size_m = float(first_voxel_size_m)
        self.requested_second_voxel_size_m = second_voxel_size_m
        self.second_candidate_ratio = float(second_candidate_ratio)
        self.second_auto_dimension = float(second_auto_dimension)
        self.second_auto_iterations = int(second_auto_iterations)
        self.second_auto_tolerance = float(second_auto_tolerance)
        self.fps_sampler = fps_sampler
        self.target_candidate_count = max(
            self.fps_sampler.num_points,
            int(
                np.ceil(
                    self.second_candidate_ratio
                    * self.fps_sampler.num_points
                )
            ),
        )

        self.first_voxelizer = GpuVoxelDownsampler(
            self.first_voxel_size_m
        )
        self.second_voxel_size_m: float | None = second_voxel_size_m
        self.second_voxelizer = (
            GpuVoxelDownsampler(second_voxel_size_m)
            if self.enable_second_downsample
            and second_voxel_size_m is not None
            else None
        )
        self.second_auto_bypass = False
        self.calibration_report: dict[str, object] | None = None

        self._host = torch.empty(
            (self.raw_point_count, 3),
            dtype=torch.float32,
            pin_memory=True,
        )
        self._host_np = self._host.numpy()
        self._device_raw = torch.empty(
            (self.raw_point_count, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._raw_indices = torch.arange(
            self.raw_point_count,
            device=self.device,
            dtype=torch.long,
        )

    def _first_stage(self) -> tuple[torch.Tensor, torch.Tensor]:
        first_points, first_local_indices = self.first_voxelizer(
            self._device_raw
        )
        first_raw_indices = self._raw_indices.index_select(
            0,
            first_local_indices,
        )
        return first_points, first_raw_indices

    def calibrate(self, points: np.ndarray) -> dict[str, object]:
        """Calibrate automatic stage two once; calibration is untimed."""
        array = np.asarray(points, dtype=np.float32)
        if array.shape != (self.raw_point_count, 3):
            raise ValueError(
                f"Expected calibration cloud {(self.raw_point_count, 3)}, "
                f"got {array.shape}."
            )
        np.copyto(self._host_np, array)
        self._device_raw.copy_(self._host, non_blocking=True)
        torch.cuda.synchronize(cuda_index(self.device))

        first_points, _ = self._first_stage()
        torch.cuda.synchronize(cuda_index(self.device))
        first_count = int(first_points.shape[0])

        report: dict[str, object] = {
            "first_count": first_count,
            "second_enabled": self.enable_second_downsample,
            "target_candidate_count": self.target_candidate_count,
            "mode": "disabled",
            "trials": [],
        }

        if not self.enable_second_downsample:
            self.second_voxel_size_m = None
            self.second_voxelizer = None
            report["candidate_count"] = first_count
            self.calibration_report = report
            return report

        if self.requested_second_voxel_size_m is not None:
            assert self.second_voxelizer is not None
            candidate_points, _ = self.second_voxelizer(first_points)
            torch.cuda.synchronize(cuda_index(self.device))
            report.update(
                {
                    "mode": "fixed",
                    "second_voxel_resolution_m": self.second_voxel_size_m,
                    "candidate_count": int(candidate_points.shape[0]),
                }
            )
            self.calibration_report = report
            return report

        if first_count <= self.target_candidate_count:
            self.second_auto_bypass = True
            self.second_voxel_size_m = None
            self.second_voxelizer = None
            report.update(
                {
                    "mode": "auto-bypass",
                    "candidate_count": first_count,
                    "second_voxel_resolution_m": None,
                }
            )
            self.calibration_report = report
            return report

        base_resolution = self.first_voxel_size_m
        initial_resolution = base_resolution * (
            first_count / self.target_candidate_count
        ) ** (1.0 / self.second_auto_dimension)
        minimum_resolution = base_resolution * 1.001
        maximum_resolution = base_resolution * 8.0
        resolution = float(
            np.clip(
                initial_resolution,
                minimum_resolution,
                maximum_resolution,
            )
        )

        best_resolution: float | None = None
        best_count: int | None = None
        best_score = float("inf")
        trials: list[dict[str, float | int]] = []

        for _ in range(self.second_auto_iterations):
            voxelizer = GpuVoxelDownsampler(resolution)
            candidate_points, _ = voxelizer(first_points)
            torch.cuda.synchronize(cuda_index(self.device))
            count = int(candidate_points.shape[0])
            trials.append(
                {
                    "resolution_m": float(resolution),
                    "candidate_count": count,
                }
            )

            if count >= self.fps_sampler.num_points:
                score = abs(
                    np.log(max(count, 1) / self.target_candidate_count)
                )
                if score < best_score:
                    best_score = float(score)
                    best_resolution = float(resolution)
                    best_count = count

            relative_error = abs(
                count - self.target_candidate_count
            ) / self.target_candidate_count
            if (
                count >= self.fps_sampler.num_points
                and relative_error <= self.second_auto_tolerance
            ):
                break

            scale = (
                max(count, 1) / self.target_candidate_count
            ) ** (1.0 / self.second_auto_dimension)
            scale = float(np.clip(scale, 0.70, 1.50))
            new_resolution = float(
                np.clip(
                    resolution * scale,
                    minimum_resolution,
                    maximum_resolution,
                )
            )
            if np.isclose(
                new_resolution,
                resolution,
                rtol=0.0,
                atol=1.0e-6,
            ):
                break
            resolution = new_resolution

        if best_resolution is None:
            best_resolution = float(minimum_resolution)
            fallback = GpuVoxelDownsampler(best_resolution)
            fallback_points, _ = fallback(first_points)
            torch.cuda.synchronize(cuda_index(self.device))
            best_count = int(fallback_points.shape[0])
            trials.append(
                {
                    "resolution_m": best_resolution,
                    "candidate_count": best_count,
                }
            )

        self.second_voxel_size_m = best_resolution
        self.second_voxelizer = GpuVoxelDownsampler(best_resolution)
        report.update(
            {
                "mode": "auto",
                "second_voxel_resolution_m": best_resolution,
                "candidate_count": int(best_count),
                "trials": trials,
            }
        )
        self.calibration_report = report
        return report

    def process(
        self,
        points: np.ndarray,
        timestamp_s: float,
    ) -> PreparedPointCloudFrame:
        array = np.asarray(points, dtype=np.float32)
        if array.shape != (self.raw_point_count, 3):
            raise ValueError(
                f"Expected raw cloud {(self.raw_point_count, 3)}, "
                f"got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Raw point cloud contains NaN or Inf values.")
        if (
            self.enable_second_downsample
            and self.requested_second_voxel_size_m is None
            and self.calibration_report is None
        ):
            raise RuntimeError(
                "Automatic second voxel resolution has not been calibrated."
            )

        wall_start = time.perf_counter()
        host_start = time.perf_counter()
        np.copyto(self._host_np, array)
        host_stage_ms = 1000.0 * (time.perf_counter() - host_start)

        event_start = torch.cuda.Event(enable_timing=True)
        event_h2d = torch.cuda.Event(enable_timing=True)
        event_first = torch.cuda.Event(enable_timing=True)
        event_second = torch.cuda.Event(enable_timing=True)
        event_fps = torch.cuda.Event(enable_timing=True)

        event_start.record()
        self._device_raw.copy_(self._host, non_blocking=True)
        event_h2d.record()

        first_points, first_raw_indices = self._first_stage()
        event_first.record()

        if self.second_voxelizer is not None:
            candidate_points, second_local_indices = self.second_voxelizer(
                first_points
            )
            candidate_raw_indices = first_raw_indices.index_select(
                0,
                second_local_indices,
            )
        else:
            candidate_points = first_points
            candidate_raw_indices = first_raw_indices
        event_second.record()

        anchor_points, fps_indices, unique_anchor_count = self.fps_sampler(
            candidate_points
        )
        anchor_raw_indices = candidate_raw_indices.index_select(
            0,
            fps_indices,
        )
        event_fps.record()
        event_fps.synchronize()

        h2d_ms = float(event_start.elapsed_time(event_h2d))
        first_downsample_ms = float(
            event_h2d.elapsed_time(event_first)
        )
        second_downsample_ms = float(
            event_first.elapsed_time(event_second)
        )
        fps_ms = float(event_second.elapsed_time(event_fps))
        preprocess_gpu_ms = float(event_start.elapsed_time(event_fps))
        preprocess_wall_ms = 1000.0 * (
            time.perf_counter() - wall_start
        )

        return PreparedPointCloudFrame(
            raw_points_cpu=array,
            first_downsample_points=first_points,
            first_raw_indices=first_raw_indices,
            candidate_points=candidate_points,
            candidate_raw_indices=candidate_raw_indices,
            anchor_points=anchor_points,
            anchor_raw_indices=anchor_raw_indices,
            timestamp_s=float(timestamp_s),
            raw_count=self.raw_point_count,
            first_count=int(first_points.shape[0]),
            candidate_count=int(candidate_points.shape[0]),
            unique_anchor_count=int(unique_anchor_count),
            second_downsample_enabled=(self.second_voxelizer is not None),
            first_voxel_resolution_m=self.first_voxel_size_m,
            second_voxel_resolution_m=self.second_voxel_size_m,
            host_stage_ms=host_stage_ms,
            h2d_ms=h2d_ms,
            first_downsample_ms=first_downsample_ms,
            second_downsample_ms=second_downsample_ms,
            fps_ms=fps_ms,
            preprocess_gpu_ms=preprocess_gpu_ms,
            preprocess_wall_ms=preprocess_wall_ms,
        )


class GlobalAnchorMotionRecoverer:
    """Recover first-level motion from every anchor using dense weights.

    Both flow and velocity are recovered using the same spatial weight matrix.
    Queries are chunked so no [N,K,3] displacement tensor is materialized.
    """

    def __init__(
        self,
        *,
        method: str,
        chunk_size: int,
        idw_power: float,
        idw_epsilon_m: float,
        softmax_sigma_m: float,
    ) -> None:
        if method not in {"inverse-distance", "softmax"}:
            raise ValueError(
                "--recovery-method must be inverse-distance or softmax."
            )
        if chunk_size < 1:
            raise ValueError("--recovery-chunk-size must be positive.")
        if idw_power <= 0.0:
            raise ValueError("--recovery-idw-power must be positive.")
        if idw_epsilon_m <= 0.0:
            raise ValueError("--recovery-idw-epsilon must be positive.")
        if softmax_sigma_m <= 0.0:
            raise ValueError("--recovery-softmax-sigma must be positive.")

        self.method = method
        self.chunk_size = int(chunk_size)
        self.idw_power = float(idw_power)
        self.idw_epsilon_m = float(idw_epsilon_m)
        self.softmax_sigma_m = float(softmax_sigma_m)

    @torch.inference_mode()
    def recover(
        self,
        *,
        query_points: torch.Tensor,
        anchor_points: torch.Tensor,
        anchor_flow: torch.Tensor,
        anchor_velocity: torch.Tensor,
    ) -> DenseMotionRecovery:
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError("query_points must have shape [N,3].")
        if anchor_points.ndim != 2 or anchor_points.shape[1] != 3:
            raise ValueError("anchor_points must have shape [K,3].")
        if anchor_flow.shape != anchor_points.shape:
            raise ValueError("anchor_flow must match anchor_points shape.")
        if anchor_velocity.shape != anchor_points.shape:
            raise ValueError("anchor_velocity must match anchor_points shape.")
        if not (
            query_points.device
            == anchor_points.device
            == anchor_flow.device
            == anchor_velocity.device
        ):
            raise ValueError("Recovery tensors must be on the same device.")

        queries = query_points.contiguous().float()
        anchors = anchor_points.contiguous().float()
        anchor_motion = torch.cat(
            (anchor_flow.float(), anchor_velocity.float()),
            dim=1,
        ).contiguous()

        anchor_t = anchors.transpose(0, 1).contiguous()
        anchor_norm2 = anchors.square().sum(dim=1).unsqueeze(0)
        recovered = torch.empty(
            (queries.shape[0], 6),
            device=queries.device,
            dtype=torch.float32,
        )

        for start in range(0, queries.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, queries.shape[0])
            query = queries[start:end]
            query_norm2 = query.square().sum(dim=1, keepdim=True)

            # [B,K] = ||q||^2 + ||a||^2 - 2 q a^T.
            dist2 = torch.addmm(
                query_norm2 + anchor_norm2,
                query,
                anchor_t,
                beta=1.0,
                alpha=-2.0,
            )
            dist2.clamp_min_(0.0)

            if self.method == "softmax":
                denominator = (
                    2.0
                    * self.softmax_sigma_m
                    * self.softmax_sigma_m
                )
                dist2.mul_(-1.0 / denominator)
                weights = torch.softmax(dist2, dim=1)
            else:
                # Standard inverse-distance weighting:
                # w = (sqrt(d2) + epsilon)^(-p).
                dist2.sqrt_()
                dist2.add_(self.idw_epsilon_m)
                dist2.pow_(-self.idw_power)
                dist2.div_(dist2.sum(dim=1, keepdim=True))
                weights = dist2

            recovered[start:end] = weights @ anchor_motion

        return DenseMotionRecovery(
            flow=recovered[:, :3].contiguous(),
            velocity=recovered[:, 3:].contiguous(),
        )


class DifFlow3DInference:
    """Existing streaming CUDA-Graph interface fed by external anchors."""

    required_frames = 2

    def __init__(self, config: DifFlow3DConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.num_points = int(config.num_points)
        self._cached_target_timestamp_s: float | None = None

        if self.device.type != "cuda":
            raise ValueError(
                "DifFlow3D PointNet++ operators require CUDA; "
                f"received device={self.device}."
            )
        if self.num_points < 1024:
            raise ValueError(
                "The supplied model/runner currently requires at least "
                "1024 input points."
            )
        if config.iters < 1:
            raise ValueError("--difflow-iters must be positive.")
        if config.uncertainty <= 0.0:
            raise ValueError("--difflow-uncertainty must be positive.")
        if config.cuda_graph_warmup < 1:
            raise ValueError("--cuda-graph-warmup must be positive.")

        repo_path = config.repo_path.expanduser().resolve()
        checkpoint_path = config.checkpoint_path.expanduser().resolve()
        module_path = repo_path / (
            config.model_module.replace(".", "/") + ".py"
        )
        if not module_path.is_file():
            raise FileNotFoundError(
                f"DifFlow3D model module was not found: {module_path}."
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"DifFlow3D checkpoint not found: {checkpoint_path}"
            )

        repo_string = str(repo_path)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(config.model_module)
        except Exception as error:
            raise RuntimeError(
                f"Failed to import {config.model_module}. Build the custom "
                "PointNet++ operators first."
            ) from error

        model_class = getattr(module, "PointConvBidirection", None)
        runner_class = getattr(
            module,
            "DifFlow3DStreamingCudaGraphRunner",
            None,
        )
        configure_fast = getattr(
            module,
            "configure_fast_inference",
            None,
        )
        if model_class is None:
            raise RuntimeError(
                f"{config.model_module}.PointConvBidirection was not found."
            )
        if runner_class is None:
            raise RuntimeError(
                f"{config.model_module} does not provide "
                "DifFlow3DStreamingCudaGraphRunner."
            )
        if configure_fast is not None:
            configure_fast(config.enable_tf32)

        self.model = model_class(iters=config.iters)
        raw_checkpoint: Any = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = self._strip_module_prefix(
            self._extract_state_dict(raw_checkpoint)
        )
        incompatible = self.model.load_state_dict(
            state_dict,
            strict=config.strict_checkpoint,
        )
        missing = tuple(getattr(incompatible, "missing_keys", ()))
        unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
        self.checkpoint_report = CheckpointReport(
            is_exact=not missing and not unexpected,
            missing_keys=missing,
            unexpected_keys=unexpected,
        )

        self.model.to(self.device)
        self.model.eval()
        if config.disable_bn_running_stats:
            for layer in self.model.modules():
                if isinstance(
                    layer,
                    (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d),
                ):
                    layer.track_running_stats = False

        self.runner = runner_class(
            self.model,
            batch_size=1,
            num_points=self.num_points,
            uncertainty=config.uncertainty,
            warmup=config.cuda_graph_warmup,
            enable_tf32=config.enable_tf32,
            dt_s=config.frame_dt_s,
        )

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Unsupported checkpoint format: expected a mapping, "
                f"received {type(checkpoint).__name__}."
            )
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict) and nested:
                return nested
        return checkpoint

    @staticmethod
    def _strip_module_prefix(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if state_dict and all(
            key.startswith("module.") for key in state_dict
        ):
            return {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
        return state_dict

    def reset(self) -> None:
        self._cached_target_timestamp_s = None
        self.runner.reset()

    def _stage_cuda_points(self, points: torch.Tensor) -> None:
        if points.shape != (self.num_points, 3):
            raise ValueError(
                f"DifFlow3D input must be {(self.num_points, 3)}, "
                f"got {tuple(points.shape)}."
            )
        if points.device != self.device:
            raise ValueError(
                f"DifFlow3D anchors are on {points.device}, expected "
                f"{self.device}."
            )
        self.runner.next_input.copy_(
            points.unsqueeze(0),
            non_blocking=True,
        )

    def infer(
        self,
        source_frame: PreparedPointCloudFrame,
        target_frame: PreparedPointCloudFrame,
    ) -> DifFlow3DEstimate:
        dt_s = float(target_frame.timestamp_s - source_frame.timestamp_s)
        if dt_s <= 0.0:
            raise ValueError(f"Non-increasing timestamps: dt={dt_s}.")
        if (
            self.config.max_frame_gap_s is not None
            and dt_s > self.config.max_frame_gap_s
        ):
            raise ValueError(
                f"Frame gap {dt_s:.6f}s exceeds "
                f"{self.config.max_frame_gap_s:.6f}s."
            )
        if not np.isclose(
            dt_s,
            self.config.frame_dt_s,
            rtol=1.0e-5,
            atol=1.0e-8,
        ):
            raise ValueError(
                "Captured CUDA Graph uses fixed dt="
                f"{self.config.frame_dt_s:.9f}s, received {dt_s:.9f}s."
            )

        source_is_cached = (
            self._cached_target_timestamp_s is not None
            and np.isclose(
                self._cached_target_timestamp_s,
                source_frame.timestamp_s,
                rtol=0.0,
                atol=1.0e-12,
            )
        )

        with torch.inference_mode():
            if not source_is_cached:
                self.runner.reset()
                self._stage_cuda_points(source_frame.anchor_points)
                if self.runner.replay_next() is not None:
                    raise RuntimeError(
                        "The first streaming frame must only be buffered."
                    )

            self._stage_cuda_points(target_frame.anchor_points)
            if self.runner.replay_next() is None:
                raise RuntimeError(
                    "Streaming decode did not produce a pair output."
                )

            predicted_flow = self.runner.flow()[0]
            source_points = self.runner.source_points()[0]
            warped_points = self.runner.warped_points()[0]
            velocity = self.runner.velocity()[0]

        self._cached_target_timestamp_s = float(target_frame.timestamp_s)
        return DifFlow3DEstimate(
            source_points=source_points,
            warped_points=warped_points,
            residual_flow=predicted_flow,
            velocity=velocity,
            valid_indices=source_frame.anchor_raw_indices,
            source_timestamp_s=float(source_frame.timestamp_s),
            target_timestamp_s=float(target_frame.timestamp_s),
        )


class RvizPipelinePublisher:
    """Publish preprocessing, anchor motion, and recovered first-level motion."""

    def __init__(
        self,
        *,
        frame_id: str,
        max_arrows: int,
        vector_scale: float,
        cloud_max_points: int,
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Point
            from rclpy.node import Node
            from rclpy.qos import (
                DurabilityPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from sensor_msgs.msg import PointCloud2
            from sensor_msgs_py import point_cloud2
            from std_msgs.msg import Header
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as error:
            raise RuntimeError(
                "RViz publishing requires rclpy, sensor_msgs_py, "
                "geometry_msgs, and visualization_msgs."
            ) from error

        self._rclpy = rclpy
        self._Point = Point
        self._point_cloud2 = point_cloud2
        self._Header = Header
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self.frame_id = frame_id
        self.max_arrows = int(max_arrows)
        self.vector_scale = float(vector_scale)
        self.cloud_max_points = int(cloud_max_points)

        if not rclpy.ok():
            rclpy.init(args=[])
        self.node: Node = rclpy.create_node(
            "difflow3d_dense_recovery_visualizer"
        )
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.raw_pub = self.node.create_publisher(
            PointCloud2,
            "/pipeline/raw_target",
            qos,
        )
        self.first_target_pub = self.node.create_publisher(
            PointCloud2,
            "/pipeline/first_downsample_target",
            qos,
        )
        self.second_target_pub = self.node.create_publisher(
            PointCloud2,
            "/pipeline/second_downsample_target",
            qos,
        )
        self.anchor_source_pub = self.node.create_publisher(
            PointCloud2,
            "/pipeline/anchor_source",
            qos,
        )
        self.anchor_target_pub = self.node.create_publisher(
            PointCloud2,
            "/pipeline/anchor_target",
            qos,
        )
        self.anchor_warped_pub = self.node.create_publisher(
            PointCloud2,
            "/pipeline/anchor_predicted_warped",
            qos,
        )
        self.first_source_pub = self.node.create_publisher(
            PointCloud2,
            "/pipeline/recovered_first_source",
            qos,
        )
        self.first_warped_pub = self.node.create_publisher(
            PointCloud2,
            "/pipeline/recovered_first_warped",
            qos,
        )
        self.anchor_vectors_pub = self.node.create_publisher(
            MarkerArray,
            "/pipeline/anchor_displacement_vectors",
            qos,
        )
        self.first_vectors_pub = self.node.create_publisher(
            MarkerArray,
            "/pipeline/recovered_first_displacement_vectors",
            qos,
        )

    def _header(self):
        header = self._Header()
        header.frame_id = self.frame_id
        header.stamp = self.node.get_clock().now().to_msg()
        return header

    def _cloud(self, points: np.ndarray):
        xyz = np.ascontiguousarray(points, dtype=np.float32)
        return self._point_cloud2.create_cloud_xyz32(
            self._header(),
            xyz,
        )

    @staticmethod
    def _uniform_indices(count: int, limit: int) -> np.ndarray:
        if limit <= 0 or count <= limit:
            return np.arange(count, dtype=np.int64)
        return np.linspace(0, count - 1, limit, dtype=np.int64)

    def _limited_tensor(
        self,
        points: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = self._uniform_indices(
            int(points.shape[0]),
            self.cloud_max_points,
        )
        if indices.size == int(points.shape[0]):
            return points.detach().cpu().numpy(), indices
        index_tensor = torch.from_numpy(indices).to(
            points.device,
            dtype=torch.long,
        )
        return (
            points.index_select(0, index_tensor).detach().cpu().numpy(),
            indices,
        )

    def _flow_marker(
        self,
        *,
        marker_id: int,
        namespace: str,
        source_points: np.ndarray,
        flow: np.ndarray,
        rgb: tuple[float, float, float],
        line_width: float,
    ):
        marker = self._Marker()
        marker.header = self._header()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = self._Marker.LINE_LIST
        marker.action = self._Marker.ADD
        marker.scale.x = float(line_width)
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 0.9
        marker.pose.orientation.w = 1.0

        indices = self._uniform_indices(
            source_points.shape[0],
            self.max_arrows,
        )
        for index in indices:
            start = source_points[index]
            end = start + self.vector_scale * flow[index]
            point_start = self._Point()
            point_start.x, point_start.y, point_start.z = map(float, start)
            point_end = self._Point()
            point_end.x, point_end.y, point_end.z = map(float, end)
            marker.points.extend((point_start, point_end))
        return marker

    def _publish_target_preprocess(
        self,
        frame: PreparedPointCloudFrame,
    ) -> None:
        raw_indices = self._uniform_indices(
            frame.raw_count,
            self.cloud_max_points,
        )
        self.raw_pub.publish(
            self._cloud(frame.raw_points_cpu[raw_indices])
        )
        first_target, _ = self._limited_tensor(
            frame.first_downsample_points
        )
        second_target, _ = self._limited_tensor(frame.candidate_points)
        self.first_target_pub.publish(self._cloud(first_target))
        self.second_target_pub.publish(self._cloud(second_target))
        self.anchor_target_pub.publish(
            self._cloud(frame.anchor_points.detach().cpu().numpy())
        )

    def publish_buffered(self, frame: PreparedPointCloudFrame) -> None:
        self._publish_target_preprocess(frame)
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_pair(
        self,
        *,
        source_frame: PreparedPointCloudFrame,
        target_frame: PreparedPointCloudFrame,
        estimate: DifFlow3DEstimate,
        recovery: DenseMotionRecovery,
        anchor_gt_flow: np.ndarray,
        first_gt_flow: np.ndarray,
    ) -> None:
        self._publish_target_preprocess(target_frame)

        anchor_source = estimate.source_points.detach().cpu().numpy()
        anchor_flow = estimate.residual_flow.detach().cpu().numpy()
        anchor_warped = estimate.warped_points.detach().cpu().numpy()
        first_source = (
            source_frame.first_downsample_points.detach().cpu().numpy()
        )
        first_flow = recovery.flow.detach().cpu().numpy()
        first_warped = first_source + first_flow

        self.anchor_source_pub.publish(self._cloud(anchor_source))
        self.anchor_warped_pub.publish(self._cloud(anchor_warped))
        self.first_source_pub.publish(self._cloud(first_source))
        self.first_warped_pub.publish(self._cloud(first_warped))

        anchor_markers = self._MarkerArray()
        anchor_markers.markers.append(
            self._flow_marker(
                marker_id=0,
                namespace="anchor_predicted",
                source_points=anchor_source,
                flow=anchor_flow,
                rgb=(1.0, 0.15, 0.15),
                line_width=0.004,
            )
        )
        anchor_markers.markers.append(
            self._flow_marker(
                marker_id=1,
                namespace="anchor_ground_truth",
                source_points=anchor_source,
                flow=anchor_gt_flow,
                rgb=(0.15, 1.0, 0.20),
                line_width=0.0025,
            )
        )
        self.anchor_vectors_pub.publish(anchor_markers)

        first_markers = self._MarkerArray()
        first_markers.markers.append(
            self._flow_marker(
                marker_id=0,
                namespace="first_recovered",
                source_points=first_source,
                flow=first_flow,
                rgb=(0.20, 0.45, 1.0),
                line_width=0.003,
            )
        )
        first_markers.markers.append(
            self._flow_marker(
                marker_id=1,
                namespace="first_ground_truth",
                source_points=first_source,
                flow=first_gt_flow,
                rgb=(0.15, 1.0, 0.20),
                line_width=0.002,
            )
        )
        self.first_vectors_pub.publish(first_markers)
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def hold(self, seconds: float) -> None:
        if seconds == 0.0:
            return
        if seconds < 0.0:
            try:
                while self._rclpy.ok():
                    self._rclpy.spin_once(
                        self.node,
                        timeout_sec=0.1,
                    )
            except KeyboardInterrupt:
                pass
            return
        deadline = time.perf_counter() + seconds
        while self._rclpy.ok() and time.perf_counter() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": percentile(values, 95.0),
        "max": float(values.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark raw -> mandatory voxel-1 -> optional voxel-2 -> FPS "
            "-> DifFlow3D -> dense first-level motion recovery."
        )
    )
    parser.add_argument("--difflow-repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-module", default="model_difflow")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--cuda-graph-warmup", type=int, default=10)
    parser.add_argument(
        "--fps-points",
        "--difflow-num-points",
        dest="fps_points",
        type=int,
        default=2048,
        help="Final anchor count and fixed DifFlow3D input size.",
    )
    parser.add_argument(
        "--fps-shortfall",
        choices=("error", "repeat"),
        default="error",
    )
    parser.add_argument("--difflow-iters", type=int, default=4)
    parser.add_argument("--difflow-uncertainty", type=float, default=0.2)
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument("--keep-bn-running-stats", action="store_true")

    parser.add_argument("--all-points", type=int, default=300000)
    parser.add_argument(
        "--voxel-resolution",
        type=float,
        default=0.010,
        help="Mandatory first voxel side length in metres.",
    )

    parser.add_argument(
        "--enable-second-downsample",
        action="store_true",
        help=(
            "Enable adaptive/fixed second voxel candidate reduction. "
            "It is disabled by default."
        ),
    )
    parser.add_argument(
        "--second-voxel-resolution",
        type=float,
        default=0.0,
        help="Fixed second voxel size; 0 selects automatic calibration.",
    )
    parser.add_argument(
        "--second-candidate-ratio",
        type=float,
        default=4.0,
        help="Auto stage two targets this multiple of --fps-points.",
    )
    parser.add_argument("--second-auto-dimension", type=float, default=2.0)
    parser.add_argument("--second-auto-iterations", type=int, default=8)
    parser.add_argument("--second-auto-tolerance", type=float, default=0.10)

    parser.add_argument(
        "--recovery-method",
        choices=("inverse-distance", "softmax"),
        default="softmax",
        help="All-anchor weighting used to recover first-level motion.",
    )
    parser.add_argument(
        "--recovery-chunk-size",
        type=int,
        default=4096,
        help="First-level queries per dense distance chunk.",
    )
    parser.add_argument(
        "--recovery-idw-power",
        type=float,
        default=2.0,
        help="Power p in inverse-distance weighting.",
    )
    parser.add_argument(
        "--recovery-idw-epsilon",
        type=float,
        default=1.0e-5,
        help="Distance regularizer in metres for inverse-distance weighting.",
    )
    parser.add_argument(
        "--recovery-softmax-sigma",
        type=float,
        default=0.025,
        help="Gaussian softmax bandwidth in metres.",
    )

    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--sensor-hz", type=float, default=30.0)
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Untimed preprocess/model/recovery warmup repetitions.",
    )
    parser.add_argument("--mesh-resolution", type=int, default=80)
    parser.add_argument(
        "--sensor-noise-std",
        type=float,
        default=DEFAULT_SENSOR_NOISE_STD_M,
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--same-samples-across-frames",
        dest="same_samples_across_frames",
        action="store_true",
        help="Reuse material surface samples in every frame (default).",
    )
    sampling_group.add_argument(
        "--independent-samples-across-frames",
        dest="same_samples_across_frames",
        action="store_false",
        help="Resample each object's observed surface every frame.",
    )
    parser.set_defaults(same_samples_across_frames=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--last-frame-npz", type=Path, default=None)

    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--rviz-frame-id", default="world")
    parser.add_argument("--rviz-max-arrows", type=int, default=1024)
    parser.add_argument("--rviz-vector-scale", type=float, default=10.0)
    parser.add_argument(
        "--rviz-cloud-max-points",
        type=int,
        default=100000,
        help="Visual-only cap for each published point cloud.",
    )
    parser.add_argument(
        "--rviz-publish-every",
        type=int,
        default=1,
        help="Publish one RViz update every N input frames.",
    )
    parser.add_argument("--rviz-hold-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sensor_hz <= 0.0:
        raise ValueError("--sensor-hz must be positive.")
    if args.all_points < 1:
        raise ValueError("--all-points must be positive.")
    if args.voxel_resolution <= 0.0:
        raise ValueError("--voxel-resolution must be positive.")
    if args.second_voxel_resolution < 0.0:
        raise ValueError("--second-voxel-resolution cannot be negative.")
    if args.rviz_publish_every < 1:
        raise ValueError("--rviz-publish-every must be positive.")

    dt_s = 1.0 / args.sensor_hz
    period_ms = 1000.0 * dt_s
    repo_path = args.difflow_repo.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else repo_path
        / "pretrain_weights"
        / "model_difflow_355_0.0114.pth"
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    torch.cuda.set_device(cuda_index(device))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_index(device))

    print("Preparing online synthetic scene generator ...")
    scene_generator = OnlineSceneGenerator(
        frame_count=args.frames,
        dt_s=dt_s,
        mesh_resolution=args.mesh_resolution,
        total_points=args.all_points,
        sensor_noise_std_m=args.sensor_noise_std,
        same_samples_across_frames=args.same_samples_across_frames,
        seed=args.seed,
    )
    calibration_scene = scene_generator.frame(0)

    fps_sampler = GpuFarthestPointSampler(
        repo_path=repo_path,
        num_points=args.fps_points,
        shortfall_policy=args.fps_shortfall,
    )
    fixed_second_resolution = (
        args.second_voxel_resolution
        if args.second_voxel_resolution > 0.0
        else None
    )
    preprocessor = FramePreprocessor(
        raw_point_count=args.all_points,
        device=device,
        first_voxel_size_m=args.voxel_resolution,
        enable_second_downsample=args.enable_second_downsample,
        second_voxel_size_m=fixed_second_resolution,
        second_candidate_ratio=args.second_candidate_ratio,
        second_auto_dimension=args.second_auto_dimension,
        second_auto_iterations=args.second_auto_iterations,
        second_auto_tolerance=args.second_auto_tolerance,
        fps_sampler=fps_sampler,
    )

    print("Calibrating optional second downsampling stage (not timed) ...")
    calibration = preprocessor.calibrate(calibration_scene.points)

    recoverer = GlobalAnchorMotionRecoverer(
        method=args.recovery_method,
        chunk_size=args.recovery_chunk_size,
        idw_power=args.recovery_idw_power,
        idw_epsilon_m=args.recovery_idw_epsilon,
        softmax_sigma_m=args.recovery_softmax_sigma,
    )

    config = DifFlow3DConfig(
        repo_path=repo_path,
        checkpoint_path=checkpoint,
        model_module=args.model_module,
        enable_tf32=not args.disable_tf32,
        cuda_graph_warmup=args.cuda_graph_warmup,
        device=args.device,
        num_points=args.fps_points,
        iters=args.difflow_iters,
        uncertainty=args.difflow_uncertainty,
        strict_checkpoint=not args.non_strict_checkpoint,
        disable_bn_running_stats=not args.keep_bn_running_stats,
        frame_dt_s=dt_s,
        max_frame_gap_s=2.0 * dt_s,
    )

    synchronize(device)
    load_start = time.perf_counter()
    estimator = DifFlow3DInference(config)
    synchronize(device)
    model_load_ms = 1000.0 * (time.perf_counter() - load_start)

    print("=" * 104)
    print("Voxel -> optional voxel-2 -> FPS -> DifFlow3D -> dense recovery")
    print("=" * 104)
    print(f"Raw points/frame:             {args.all_points}")
    print(
        f"First voxel resolution:       {args.voxel_resolution:.6f} m "
        f"({1000.0 * args.voxel_resolution:.2f} mm)"
    )
    print(
        f"Second downsampling:          "
        f"{'enabled' if args.enable_second_downsample else 'disabled'}"
    )
    second_resolution = calibration.get("second_voxel_resolution_m")
    if second_resolution is None:
        second_resolution_text = "bypassed"
    else:
        second_resolution_text = (
            f"{float(second_resolution):.6f} m "
            f"({1000.0 * float(second_resolution):.2f} mm)"
        )
    print(f"Second mode/resolution:       {calibration['mode']} / {second_resolution_text}")
    print(
        f"Calibration first/candidates: {calibration['first_count']} / "
        f"{calibration.get('candidate_count', calibration['first_count'])}"
    )
    print(f"FPS / model points:           {args.fps_points}")
    print(f"Recovery method:              {args.recovery_method}")
    print(f"Recovery chunk size:          {args.recovery_chunk_size}")
    if args.recovery_method == "softmax":
        print(
            f"Recovery softmax sigma:       "
            f"{1000.0 * args.recovery_softmax_sigma:.3f} mm"
        )
    else:
        print(f"Recovery IDW power:           {args.recovery_idw_power:.3f}")
        print(
            f"Recovery IDW epsilon:         "
            f"{1000.0 * args.recovery_idw_epsilon:.6f} mm"
        )
    print(f"Frames / sensor rate:         {args.frames} / {args.sensor_hz:.3f} Hz")
    print(f"Sensor period:                {period_ms:.3f} ms")
    print(f"DifFlow iterations:           {args.difflow_iters}")
    print(f"Model load time:              {model_load_ms:.1f} ms")
    print(
        f"Exact checkpoint:             "
        f"{estimator.checkpoint_report.is_exact}"
    )

    warm_scene_source = scene_generator.frame(0)
    warm_scene_target = scene_generator.frame(1)
    for _ in range(max(0, args.warmup)):
        estimator.reset()
        warm_source = preprocessor.process(
            warm_scene_source.points,
            warm_scene_source.timestamp_s,
        )
        warm_target = preprocessor.process(
            warm_scene_target.points,
            warm_scene_target.timestamp_s,
        )
        warm_estimate = estimator.infer(warm_source, warm_target)
        _ = recoverer.recover(
            query_points=warm_source.first_downsample_points,
            anchor_points=warm_estimate.source_points,
            anchor_flow=warm_estimate.residual_flow,
            anchor_velocity=warm_estimate.velocity,
        )
        synchronize(device)
    estimator.reset()
    torch.cuda.reset_peak_memory_stats(cuda_index(device))

    rviz = (
        RvizPipelinePublisher(
            frame_id=args.rviz_frame_id,
            max_arrows=args.rviz_max_arrows,
            vector_scale=args.rviz_vector_scale,
            cloud_max_points=args.rviz_cloud_max_points,
        )
        if args.rviz
        else None
    )

    timing: dict[str, list[float]] = {
        "scene_generation_ms": [],
        "host_stage_ms": [],
        "h2d_ms": [],
        "first_downsample_ms": [],
        "second_downsample_ms": [],
        "fps_ms": [],
        "preprocess_gpu_ms": [],
        "preprocess_wall_ms": [],
        "model_ms": [],
        "recovery_ms": [],
        "model_recovery_gpu_ms": [],
        "overall_wall_ms": [],
    }
    first_counts: list[int] = []
    candidate_counts: list[int] = []
    unique_anchor_counts: list[int] = []

    anchor_flow_epe_chunks: list[np.ndarray] = []
    anchor_average_velocity_epe_chunks: list[np.ndarray] = []
    anchor_target_velocity_epe_chunks: list[np.ndarray] = []
    first_flow_epe_chunks: list[np.ndarray] = []
    first_average_velocity_epe_chunks: list[np.ndarray] = []
    first_target_velocity_epe_chunks: list[np.ndarray] = []

    per_frame: list[dict[str, object]] = []
    previous_prepared: PreparedPointCloudFrame | None = None
    previous_scene: OnlineSceneFrame | None = None
    last_payload: dict[str, np.ndarray] | None = None

    stream_wall_start = time.perf_counter()
    print("\nStreaming frames")
    print("-" * 104)

    for target_index in range(args.frames):
        if args.realtime:
            release_time = stream_wall_start + target_index * dt_s
            sleep_s = release_time - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        scene_start = time.perf_counter()
        scene_frame = scene_generator.frame(target_index)
        scene_generation_ms = 1000.0 * (
            time.perf_counter() - scene_start
        )
        timing["scene_generation_ms"].append(scene_generation_ms)

        # Synthetic generation is sensor acquisition and is outside this timer.
        cycle_start = time.perf_counter()
        prepared = preprocessor.process(
            scene_frame.points,
            scene_frame.timestamp_s,
        )
        first_counts.append(prepared.first_count)
        candidate_counts.append(prepared.candidate_count)
        unique_anchor_counts.append(prepared.unique_anchor_count)
        for key in (
            "host_stage_ms",
            "h2d_ms",
            "first_downsample_ms",
            "second_downsample_ms",
            "fps_ms",
            "preprocess_gpu_ms",
            "preprocess_wall_ms",
        ):
            timing[key].append(float(getattr(prepared, key)))

        if previous_prepared is None or previous_scene is None:
            previous_prepared = prepared
            previous_scene = scene_frame
            if rviz is not None:
                rviz.publish_buffered(prepared)
            print(
                f"frame {target_index:03d}: raw={prepared.raw_count:7d} -> "
                f"first={prepared.first_count:7d} -> "
                f"cand={prepared.candidate_count:7d} -> "
                f"fps={args.fps_points:5d}  buffering"
            )
            continue

        model_start_event = torch.cuda.Event(enable_timing=True)
        model_end_event = torch.cuda.Event(enable_timing=True)
        recovery_start_event = torch.cuda.Event(enable_timing=True)
        recovery_end_event = torch.cuda.Event(enable_timing=True)

        model_start_event.record()
        estimate = estimator.infer(previous_prepared, prepared)
        model_end_event.record()

        recovery_start_event.record()
        recovery = recoverer.recover(
            query_points=previous_prepared.first_downsample_points,
            anchor_points=estimate.source_points,
            anchor_flow=estimate.residual_flow,
            anchor_velocity=estimate.velocity,
        )
        recovery_end_event.record()
        recovery_end_event.synchronize()

        model_ms = float(
            model_start_event.elapsed_time(model_end_event)
        )
        recovery_ms = float(
            recovery_start_event.elapsed_time(recovery_end_event)
        )
        model_recovery_gpu_ms = float(
            model_start_event.elapsed_time(recovery_end_event)
        )
        overall_wall_ms = 1000.0 * (
            time.perf_counter() - cycle_start
        )
        timing["model_ms"].append(model_ms)
        timing["recovery_ms"].append(recovery_ms)
        timing["model_recovery_gpu_ms"].append(model_recovery_gpu_ms)
        timing["overall_wall_ms"].append(overall_wall_ms)

        source_index = target_index - 1
        anchor_raw_indices = (
            estimate.valid_indices.detach().cpu().numpy().astype(np.int64)
        )
        first_raw_indices = (
            previous_prepared.first_raw_indices
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        anchor_predicted_flow = (
            estimate.residual_flow.detach().cpu().numpy()
        )
        anchor_predicted_velocity = (
            estimate.velocity.detach().cpu().numpy()
        )
        first_recovered_flow = recovery.flow.detach().cpu().numpy()
        first_recovered_velocity = recovery.velocity.detach().cpu().numpy()

        anchor_gt_flow = previous_scene.gt_flow_to_next[
            anchor_raw_indices
        ]
        anchor_gt_velocity_average = anchor_gt_flow / np.float32(dt_s)
        anchor_gt_velocity_target = previous_scene.gt_velocity_target[
            anchor_raw_indices
        ]

        first_gt_flow = previous_scene.gt_flow_to_next[first_raw_indices]
        first_gt_velocity_average = first_gt_flow / np.float32(dt_s)
        first_gt_velocity_target = previous_scene.gt_velocity_target[
            first_raw_indices
        ]

        anchor_flow_epe = np.linalg.norm(
            anchor_predicted_flow - anchor_gt_flow,
            axis=1,
        )
        anchor_average_velocity_epe = np.linalg.norm(
            anchor_predicted_velocity - anchor_gt_velocity_average,
            axis=1,
        )
        anchor_target_velocity_epe = np.linalg.norm(
            anchor_predicted_velocity - anchor_gt_velocity_target,
            axis=1,
        )
        first_flow_epe = np.linalg.norm(
            first_recovered_flow - first_gt_flow,
            axis=1,
        )
        first_average_velocity_epe = np.linalg.norm(
            first_recovered_velocity - first_gt_velocity_average,
            axis=1,
        )
        first_target_velocity_epe = np.linalg.norm(
            first_recovered_velocity - first_gt_velocity_target,
            axis=1,
        )

        anchor_flow_epe_chunks.append(anchor_flow_epe)
        anchor_average_velocity_epe_chunks.append(
            anchor_average_velocity_epe
        )
        anchor_target_velocity_epe_chunks.append(
            anchor_target_velocity_epe
        )
        first_flow_epe_chunks.append(first_flow_epe)
        first_average_velocity_epe_chunks.append(
            first_average_velocity_epe
        )
        first_target_velocity_epe_chunks.append(
            first_target_velocity_epe
        )

        row = {
            "source_index": source_index,
            "target_index": target_index,
            "raw_points": prepared.raw_count,
            "first_points": prepared.first_count,
            "candidate_points": prepared.candidate_count,
            "fps_points": args.fps_points,
            "recovery_method": args.recovery_method,
            "scene_generation_ms_outside_cycle": scene_generation_ms,
            "host_stage_ms": prepared.host_stage_ms,
            "h2d_ms": prepared.h2d_ms,
            "first_downsample_ms": prepared.first_downsample_ms,
            "second_downsample_ms": prepared.second_downsample_ms,
            "fps_ms": prepared.fps_ms,
            "preprocess_gpu_ms": prepared.preprocess_gpu_ms,
            "preprocess_wall_ms": prepared.preprocess_wall_ms,
            "model_ms": model_ms,
            "recovery_ms": recovery_ms,
            "model_recovery_gpu_ms": model_recovery_gpu_ms,
            "overall_wall_ms": overall_wall_ms,
            "anchor_mean_epe_m": float(anchor_flow_epe.mean()),
            "anchor_mean_average_velocity_epe_mps": float(
                anchor_average_velocity_epe.mean()
            ),
            "anchor_mean_target_velocity_epe_mps": float(
                anchor_target_velocity_epe.mean()
            ),
            "first_mean_epe_m": float(first_flow_epe.mean()),
            "first_mean_average_velocity_epe_mps": float(
                first_average_velocity_epe.mean()
            ),
            "first_mean_target_velocity_epe_mps": float(
                first_target_velocity_epe.mean()
            ),
            "deadline_miss": bool(overall_wall_ms > period_ms),
        }
        per_frame.append(row)

        print(
            f"{source_index:03d}->{target_index:03d} | "
            f"raw {prepared.raw_count:7d} -> "
            f"first {prepared.first_count:6d} -> "
            f"cand {prepared.candidate_count:6d} -> "
            f"FPS {args.fps_points:5d} | "
            f"V1 {prepared.first_downsample_ms:5.2f}  "
            f"V2 {prepared.second_downsample_ms:5.2f}  "
            f"FPS {prepared.fps_ms:6.2f}  "
            f"model {model_ms:6.2f}  "
            f"recover {recovery_ms:6.2f}  "
            f"overall {overall_wall_ms:7.2f} ms | "
            f"anchor EPE {anchor_flow_epe.mean():.5f}  "
            f"first EPE {first_flow_epe.mean():.5f} m"
        )

        last_payload = {
            "raw_target": prepared.raw_points_cpu,
            "first_source": (
                previous_prepared.first_downsample_points
                .detach()
                .cpu()
                .numpy()
            ),
            "first_target": (
                prepared.first_downsample_points.detach().cpu().numpy()
            ),
            "candidate_target": (
                prepared.candidate_points.detach().cpu().numpy()
            ),
            "anchor_source": estimate.source_points.detach().cpu().numpy(),
            "anchor_target": prepared.anchor_points.detach().cpu().numpy(),
            "anchor_predicted_warped": (
                estimate.warped_points.detach().cpu().numpy()
            ),
            "anchor_predicted_flow": anchor_predicted_flow,
            "anchor_predicted_velocity": anchor_predicted_velocity,
            "anchor_gt_flow": anchor_gt_flow,
            "first_recovered_flow": first_recovered_flow,
            "first_recovered_velocity": first_recovered_velocity,
            "first_recovered_warped": (
                previous_prepared.first_downsample_points
                .detach()
                .cpu()
                .numpy()
                + first_recovered_flow
            ),
            "first_gt_flow": first_gt_flow,
            "anchor_raw_indices": anchor_raw_indices,
            "first_raw_indices": first_raw_indices,
        }

        if (
            rviz is not None
            and target_index % args.rviz_publish_every == 0
        ):
            rviz.publish_pair(
                source_frame=previous_prepared,
                target_frame=prepared,
                estimate=estimate,
                recovery=recovery,
                anchor_gt_flow=anchor_gt_flow,
                first_gt_flow=first_gt_flow,
            )

        previous_prepared = prepared
        previous_scene = scene_frame

    if not timing["overall_wall_ms"]:
        raise RuntimeError("No DifFlow3D pair was produced.")

    timing_summary = {
        key: summarize(values)
        for key, values in timing.items()
    }
    first_array = np.asarray(first_counts, dtype=np.float64)
    candidate_array = np.asarray(candidate_counts, dtype=np.float64)
    unique_anchor_array = np.asarray(
        unique_anchor_counts,
        dtype=np.float64,
    )
    overall_array = np.asarray(
        timing["overall_wall_ms"],
        dtype=np.float64,
    )

    anchor_flow_epe_all = np.concatenate(anchor_flow_epe_chunks)
    anchor_avg_vel_epe_all = np.concatenate(
        anchor_average_velocity_epe_chunks
    )
    anchor_target_vel_epe_all = np.concatenate(
        anchor_target_velocity_epe_chunks
    )
    first_flow_epe_all = np.concatenate(first_flow_epe_chunks)
    first_avg_vel_epe_all = np.concatenate(
        first_average_velocity_epe_chunks
    )
    first_target_vel_epe_all = np.concatenate(
        first_target_velocity_epe_chunks
    )

    accuracy = {
        "anchor_level": {
            "flow_epe_m": metric_summary(anchor_flow_epe_all),
            "average_velocity_epe_mps": metric_summary(
                anchor_avg_vel_epe_all
            ),
            "target_velocity_epe_mps": metric_summary(
                anchor_target_vel_epe_all
            ),
        },
        "first_level_recovered": {
            "flow_epe_m": metric_summary(first_flow_epe_all),
            "average_velocity_epe_mps": metric_summary(
                first_avg_vel_epe_all
            ),
            "target_velocity_epe_mps": metric_summary(
                first_target_vel_epe_all
            ),
        },
    }

    result: dict[str, object] = {
        "pipeline": {
            "raw_points": args.all_points,
            "first_voxel_resolution_m": args.voxel_resolution,
            "second_downsample_enabled": args.enable_second_downsample,
            "second_mode": calibration["mode"],
            "second_voxel_resolution_m": (
                preprocessor.second_voxel_size_m
            ),
            "second_candidate_ratio_target": (
                args.second_candidate_ratio
            ),
            "target_candidate_points": (
                preprocessor.target_candidate_count
            ),
            "fps_points": args.fps_points,
            "recovery": {
                "method": args.recovery_method,
                "chunk_size": args.recovery_chunk_size,
                "idw_power": args.recovery_idw_power,
                "idw_epsilon_m": args.recovery_idw_epsilon,
                "softmax_sigma_m": args.recovery_softmax_sigma,
            },
            "mean_first_points": float(first_array.mean()),
            "median_first_points": float(np.median(first_array)),
            "min_first_points": int(first_array.min()),
            "max_first_points": int(first_array.max()),
            "mean_candidate_points": float(candidate_array.mean()),
            "median_candidate_points": float(
                np.median(candidate_array)
            ),
            "min_candidate_points": int(candidate_array.min()),
            "max_candidate_points": int(candidate_array.max()),
            "min_unique_fps_points": int(unique_anchor_array.min()),
            "calibration": calibration,
        },
        "model": {
            "name": "difflow3d",
            "model_module": args.model_module,
            "checkpoint": str(checkpoint),
            "iters": args.difflow_iters,
            "uncertainty": args.difflow_uncertainty,
            "tf32_enabled": not args.disable_tf32,
            "cuda_graph_warmup": args.cuda_graph_warmup,
            "load_time_ms": model_load_ms,
            "exact_checkpoint": estimator.checkpoint_report.is_exact,
        },
        "timing_ms": timing_summary,
        "realtime": {
            "sensor_hz": args.sensor_hz,
            "period_ms": period_ms,
            "deadline_miss_ratio": float(
                np.mean(overall_array > period_ms)
            ),
            "sustainable_hz_from_median": float(
                1000.0 / np.median(overall_array)
            ),
        },
        "accuracy": accuracy,
        "gpu_memory_mib": {
            "peak_allocated": float(
                torch.cuda.max_memory_allocated(cuda_index(device))
                / 1024**2
            ),
            "reserved": float(
                torch.cuda.memory_reserved(cuda_index(device)) / 1024**2
            ),
        },
        "per_frame": per_frame,
    }

    print("\n" + "=" * 104)
    print("Point-count summary")
    print("=" * 104)
    print(f"Raw points:                    {args.all_points}")
    print(
        f"First points mean/median:      {first_array.mean():.1f} / "
        f"{np.median(first_array):.1f}"
    )
    print(
        f"First points min/max:          {int(first_array.min())} / "
        f"{int(first_array.max())}"
    )
    print(
        f"Second downsampling enabled:   "
        f"{args.enable_second_downsample}"
    )
    print(f"Second stage mode:             {calibration['mode']}")
    if preprocessor.second_voxel_size_m is not None:
        print(
            f"Second resolution:             "
            f"{1000.0 * preprocessor.second_voxel_size_m:.3f} mm"
        )
    else:
        print("Second resolution:             bypassed")
    print(
        f"Candidate points mean/median:  {candidate_array.mean():.1f} / "
        f"{np.median(candidate_array):.1f}"
    )
    print(f"Final anchor points:           {args.fps_points}")

    print("\n" + "=" * 104)
    print("Cycle-time statistics")
    print("=" * 104)
    for key, label in (
        ("scene_generation_ms", "Synthetic scene (outside)"),
        ("host_stage_ms", "CPU -> pinned stage"),
        ("h2d_ms", "Pinned H2D"),
        ("first_downsample_ms", "GPU first downsample"),
        ("second_downsample_ms", "GPU second downsample"),
        ("fps_ms", "GPU FPS"),
        ("preprocess_gpu_ms", "GPU preprocess total"),
        ("preprocess_wall_ms", "Preprocess wall total"),
        ("model_ms", "DifFlow3D model"),
        ("recovery_ms", "Dense motion recovery"),
        ("model_recovery_gpu_ms", "Model + recovery GPU"),
        ("overall_wall_ms", "Online overall"),
    ):
        print_timing_row(label, timing_summary[key])

    print("\n" + "=" * 104)
    print("Anchor-level accuracy")
    print("=" * 104)
    anchor_accuracy = accuracy["anchor_level"]
    print(
        f"Flow EPE mean/median/P95:       "
        f"{anchor_accuracy['flow_epe_m']['mean']:.6f} / "
        f"{anchor_accuracy['flow_epe_m']['median']:.6f} / "
        f"{anchor_accuracy['flow_epe_m']['p95']:.6f} m"
    )
    print(
        f"Avg-velocity EPE mean/P95:      "
        f"{anchor_accuracy['average_velocity_epe_mps']['mean']:.6f} / "
        f"{anchor_accuracy['average_velocity_epe_mps']['p95']:.6f} m/s"
    )
    print(
        f"Target-velocity EPE mean/P95:   "
        f"{anchor_accuracy['target_velocity_epe_mps']['mean']:.6f} / "
        f"{anchor_accuracy['target_velocity_epe_mps']['p95']:.6f} m/s"
    )

    print("\n" + "=" * 104)
    print("Recovered first-level accuracy")
    print("=" * 104)
    first_accuracy = accuracy["first_level_recovered"]
    print(
        f"Flow EPE mean/median/P95:       "
        f"{first_accuracy['flow_epe_m']['mean']:.6f} / "
        f"{first_accuracy['flow_epe_m']['median']:.6f} / "
        f"{first_accuracy['flow_epe_m']['p95']:.6f} m"
    )
    print(
        f"Avg-velocity EPE mean/P95:      "
        f"{first_accuracy['average_velocity_epe_mps']['mean']:.6f} / "
        f"{first_accuracy['average_velocity_epe_mps']['p95']:.6f} m/s"
    )
    print(
        f"Target-velocity EPE mean/P95:   "
        f"{first_accuracy['target_velocity_epe_mps']['mean']:.6f} / "
        f"{first_accuracy['target_velocity_epe_mps']['p95']:.6f} m/s"
    )
    print(
        f"Deadline misses:                "
        f"{100.0 * np.mean(overall_array > period_ms):.2f}%"
    )
    print(
        f"Sustainable rate:               "
        f"{1000.0 / np.median(overall_array):.2f} Hz"
    )
    print(
        f"Peak GPU memory:                "
        f"{result['gpu_memory_mib']['peak_allocated']:.1f} MiB"
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved JSON report: {args.json_output}")

    if args.last_frame_npz is not None:
        if last_payload is None:
            raise RuntimeError("No final pair is available to save.")
        args.last_frame_npz.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        np.savez_compressed(args.last_frame_npz, **last_payload)
        print(f"Saved final arrays: {args.last_frame_npz}")

    if rviz is not None:
        print(
            "\nRViz topics:\n"
            "  /pipeline/raw_target\n"
            "  /pipeline/first_downsample_target\n"
            "  /pipeline/second_downsample_target\n"
            "  /pipeline/anchor_source\n"
            "  /pipeline/anchor_target\n"
            "  /pipeline/anchor_predicted_warped\n"
            "  /pipeline/recovered_first_source\n"
            "  /pipeline/recovered_first_warped\n"
            "  /pipeline/anchor_displacement_vectors\n"
            "  /pipeline/recovered_first_displacement_vectors\n"
            "Anchor predicted vectors are red; recovered first-level vectors "
            "are blue; ground truth vectors are green."
        )
        rviz.hold(args.rviz_hold_seconds)
        rviz.close()


if __name__ == "__main__":
    main()