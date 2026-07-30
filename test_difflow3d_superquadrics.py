"""Online scene-flow test using the four-superquadric safety-filter scene.

The scene matches the supplied dynamic-superquadric generator:
- main_box: static;
- left_rounded_pillar: sinusoidal translation along world Y;
- right_flat_ellipsoid: sinusoidal rotation about world Z;
- front_tilted_box: constant world-X translation.

For each source frame, ground-truth flow is evaluated on the exact sampled
source surface points by transforming those material samples to the next pose.
The target frame may independently resample each surface, matching the original
scene generator and avoiding artificial point correspondences.

This script measures:
- end-to-end online inference latency;
- flow endpoint error (EPE);
- velocity endpoint error;
- per-object accuracy;
- moving/static accuracy;
- valid-point ratio and deadline misses.

The official DifFlow3D checkpoint was trained on the FlyingThings3D subset.
This simulation is therefore a pipeline and domain-transfer test, not an
official FlyingThings3D or KITTI benchmark.

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


def generate_sequence(
    *,
    frame_count: int,
    dt_s: float,
    mesh_resolution: int,
    point_multiplier: int,
    sensor_noise_std_m: float,
    same_samples_across_frames: bool,
    seed: int,
) -> SceneSequence:
    if frame_count < 3:
        raise ValueError("--frames must be at least 3.")
    if dt_s <= 0.0:
        raise ValueError("Frame interval must be positive.")
    if point_multiplier < 1:
        raise ValueError("--point-multiplier must be at least 1.")
    if sensor_noise_std_m < 0.0:
        raise ValueError("--sensor-noise-std must be non-negative.")

    rng = np.random.default_rng(seed)
    timestamps_s = np.arange(frame_count, dtype=np.float64) * dt_s

    samplers: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    fixed_samples: dict[str, np.ndarray] = {}

    counts: list[int] = []
    for spec in OBSTACLE_SPECS:
        vertices, faces = make_superquadric_mesh(spec, mesh_resolution)
        samplers[spec.name] = prepare_triangle_sampler(vertices, faces)

        count = spec.points_per_frame * point_multiplier
        counts.append(count)

        if same_samples_across_frames:
            fixed_samples[spec.name] = sample_points_on_triangle_mesh(
                samplers[spec.name],
                count,
                rng,
            )

    total_points = int(sum(counts))
    points = np.empty(
        (frame_count, total_points, 3),
        dtype=np.float32,
    )
    gt_flow = np.empty(
        (frame_count - 1, total_points, 3),
        dtype=np.float32,
    )

    gt_velocity_target = np.empty(
        (frame_count - 1, total_points, 3),
        dtype=np.float32,
    )

    object_ids = np.concatenate(
        [
            np.full(count, object_index, dtype=np.int16)
            for object_index, count in enumerate(counts)
        ]
    )
    dynamic_mask = np.concatenate(
        [
            np.full(
                count,
                spec.motion_mode != "static",
                dtype=bool,
            )
            for spec, count in zip(OBSTACLE_SPECS, counts)
        ]
    )

    max_gt_flow_by_object_m = {
        spec.name: 0.0
        for spec in OBSTACLE_SPECS
    }

    for frame_index, time_s in enumerate(timestamps_s):
        point_offset = 0

        for spec, count in zip(OBSTACLE_SPECS, counts):
            if same_samples_across_frames:
                local_points = fixed_samples[spec.name]
            else:
                local_points = sample_points_on_triangle_mesh(
                    samplers[spec.name],
                    count,
                    rng,
                )

            # translation, rotation = pose_at_time(spec, float(time_s))

            (
                translation,
                rotation,
                linear_velocity_world,
                angular_velocity_world,
            ) = motion_state_at_time(
                spec,
                float(time_s),
            )
            
            clean_points = transform_points(
                local_points,
                translation,
                rotation,
            )

            noise = rng.normal(
                loc=0.0,
                scale=sensor_noise_std_m,
                size=clean_points.shape,
            )
            observed_points = clean_points + noise

            point_slice = slice(point_offset, point_offset + count)
            points[frame_index, point_slice] = observed_points.astype(
                np.float32
            )

            # if frame_index < frame_count - 1:
            #     next_translation, next_rotation = pose_at_time(
            #         spec,
            #         float(time_s + dt_s),
            #     )
            #     corresponding_next_points = transform_points(
            #         local_points,
            #         next_translation,
            #         next_rotation,
            #     )
            #     object_flow = corresponding_next_points - clean_points
            #     gt_flow[frame_index, point_slice] = object_flow.astype(
            #         np.float32
            #     )

            #     max_gt_flow_by_object_m[spec.name] = max(
            #         max_gt_flow_by_object_m[spec.name],
            #         float(
            #             np.linalg.norm(object_flow, axis=1).max(
            #                 initial=0.0
            #             )
            #         ),
            #     )

            if frame_index < frame_count - 1:
                target_time_s = float(time_s + dt_s)

                (
                    next_translation,
                    next_rotation,
                    next_linear_velocity_world,
                    next_angular_velocity_world,
                ) = motion_state_at_time(
                    spec,
                    target_time_s,
                )

                corresponding_next_points = transform_points(
                    local_points,
                    next_translation,
                    next_rotation,
                )

                # Exact finite-interval flow.
                object_flow = (
                    corresponding_next_points
                    - clean_points
                )

                gt_flow[
                    frame_index,
                    point_slice,
                ] = object_flow.astype(np.float32)

                max_gt_flow_by_object_m[spec.name] = max(
                    max_gt_flow_by_object_m[spec.name],
                    float(
                        np.linalg.norm(object_flow, axis=1).max(
                            initial=0.0
                        )
                    ),
                )

                # Analytic instantaneous velocity at the target time.
                object_velocity_target = (
                    analytic_point_velocity_world(
                        local_points=local_points,
                        rotation_world_object=next_rotation,
                        linear_velocity_world=(
                            next_linear_velocity_world
                        ),
                        angular_velocity_world=(
                            next_angular_velocity_world
                        ),
                    )
                )

                gt_velocity_target[
                    frame_index,
                    point_slice,
                ] = object_velocity_target.astype(
                    np.float32
                )

            point_offset += count

    # return SceneSequence(
    #     points=np.ascontiguousarray(points),
    #     timestamps_s=timestamps_s,
    #     gt_flow=np.ascontiguousarray(gt_flow),
    #     object_ids=object_ids,
    #     object_names=tuple(spec.name for spec in OBSTACLE_SPECS),
    #     dynamic_mask=dynamic_mask,
    #     max_gt_flow_by_object_m=max_gt_flow_by_object_m,
    # )

    return SceneSequence(
        points=np.ascontiguousarray(points),
        timestamps_s=timestamps_s,
        gt_flow=np.ascontiguousarray(gt_flow),
        gt_velocity_target=np.ascontiguousarray(
            gt_velocity_target
        ),
        object_ids=object_ids,
        object_names=tuple(
            spec.name for spec in OBSTACLE_SPECS
        ),
        dynamic_mask=dynamic_mask,
        max_gt_flow_by_object_m=(
            max_gt_flow_by_object_m
        ),
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


@dataclass(frozen=True)
class PointCloudFrame:
    """Minimal frame container matching the previous test script interface."""

    points: torch.Tensor
    pose_world_sensor: torch.Tensor
    timestamp_s: float


@dataclass(frozen=True)
class DifFlow3DConfig:
    """Configuration for the minimal streaming inference path."""

    repo_path: Path
    checkpoint_path: Path
    model_module: str = "model_difflow"
    enable_tf32: bool = True
    cuda_graph_warmup: int = 10
    device: str = "cuda:0"
    num_points: int = 1024
    iters: int = 2
    uncertainty: float = 0.2
    strict_checkpoint: bool = True
    disable_bn_running_stats: bool = True
    seed: int = 42
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


class DifFlow3DInference:
    """Minimal adapter for the streaming CUDA Graph deployment model.

    Each physical frame is sampled and encoded once. The sampled target of
    pair ``(t-1, t)`` is reused as the source of pair ``(t, t+1)``.
    """

    required_frames = 2

    def __init__(self, config: DifFlow3DConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.num_points = int(config.num_points)
        self._pair_counter = 0

        self._cached_target_timestamp_s: float | None = None
        self._cached_target_points: np.ndarray | None = None
        self._cached_target_indices: np.ndarray | None = None

        if self.device.type != "cuda":
            raise ValueError(
                "DifFlow3D PointNet++ operators require CUDA; "
                f"received device={self.device}."
            )
        if self.num_points < 1024:
            raise ValueError(
                "model_difflow and its streaming graph runner "
                "currently require --difflow-num-points >= 1024."
            )
        if config.iters < 1:
            raise ValueError("--difflow-iters must be positive.")
        if config.uncertainty <= 0.0:
            raise ValueError("--difflow-uncertainty must be positive.")
        if config.cuda_graph_warmup < 1:
            raise ValueError("--cuda-graph-warmup must be positive.")
        if config.frame_dt_s <= 0.0:
            raise ValueError("frame_dt_s must be positive.")

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
                "PointNet++ operators first with "
                "`cd pointnet2 && python setup.py install`."
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

        shape = (1, self.num_points, 3)
        self._source_host = torch.empty(
            shape,
            dtype=torch.float32,
            pin_memory=True,
        )
        self._target_host = torch.empty(
            shape,
            dtype=torch.float32,
            pin_memory=True,
        )
        self._source_host_np = self._source_host[0].numpy()
        self._target_host_np = self._target_host[0].numpy()

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
            key.startswith("module.")
            for key in state_dict
        ):
            return {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
        return state_dict

    def reset_sampling(self) -> None:
        self._pair_counter = 0
        self._cached_target_timestamp_s = None
        self._cached_target_points = None
        self._cached_target_indices = None
        self.runner.reset()

    def _sample_indices(
        self,
        point_count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if point_count < 1:
            raise ValueError("Cannot run on an empty point cloud.")

        return rng.choice(
            point_count,
            size=self.num_points,
            replace=point_count < self.num_points,
        ).astype(np.int64, copy=False)

    @staticmethod
    def _as_numpy_points(points: torch.Tensor) -> np.ndarray:
        array = np.asarray(
            points.detach().cpu().numpy(),
            dtype=np.float32,
        )
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(
                f"Point cloud must have shape [N,3], got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Point cloud contains NaN or Inf values.")
        return np.ascontiguousarray(array)

    def _stage_frame(
        self,
        points: np.ndarray,
        *,
        source_slot: bool,
    ) -> None:
        if source_slot:
            np.copyto(self._source_host_np, points)
            host = self._source_host
        else:
            np.copyto(self._target_host_np, points)
            host = self._target_host

        self.runner.next_input.copy_(
            host,
            non_blocking=True,
        )

    def _cached_source_matches(self, timestamp_s: float) -> bool:
        return (
            self._cached_target_timestamp_s is not None
            and np.isclose(
                self._cached_target_timestamp_s,
                timestamp_s,
                rtol=0.0,
                atol=1.0e-12,
            )
            and self._cached_target_points is not None
            and self._cached_target_indices is not None
        )

    def infer(
        self,
        frames: list[PointCloudFrame],
    ) -> DifFlow3DEstimate:
        if len(frames) != self.required_frames:
            raise ValueError(
                f"DifFlow3D requires exactly 2 frames, got {len(frames)}."
            )

        source_frame, target_frame = frames
        dt_s = float(
            target_frame.timestamp_s
            - source_frame.timestamp_s
        )
        if dt_s <= 0.0:
            raise ValueError(
                f"Frame timestamps are not increasing: dt={dt_s}."
            )
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
                "The captured CUDA Graph uses a fixed frame interval of "
                f"{self.config.frame_dt_s:.9f}s, but received {dt_s:.9f}s."
            )

        source_full = self._as_numpy_points(source_frame.points)
        target_full = self._as_numpy_points(target_frame.points)

        pair_seed = (
            self.config.seed
            + 104729 * self._pair_counter
        )
        self._pair_counter += 1

        source_is_cached = self._cached_source_matches(
            float(source_frame.timestamp_s)
        )

        if source_is_cached:
            assert self._cached_target_points is not None
            assert self._cached_target_indices is not None
            source_np = self._cached_target_points
            source_indices_np = self._cached_target_indices
        else:
            source_rng = np.random.default_rng(pair_seed)
            source_indices_np = self._sample_indices(
                source_full.shape[0],
                source_rng,
            )
            source_np = np.ascontiguousarray(
                source_full[source_indices_np],
                dtype=np.float32,
            )

        target_rng = np.random.default_rng(pair_seed + 1)
        target_indices_np = self._sample_indices(
            target_full.shape[0],
            target_rng,
        )
        target_np = np.ascontiguousarray(
            target_full[target_indices_np],
            dtype=np.float32,
        )

        with torch.inference_mode():
            if not source_is_cached:
                self.runner.reset()
                self._stage_frame(
                    source_np,
                    source_slot=True,
                )
                if self.runner.replay_next() is not None:
                    raise RuntimeError(
                        "The first streaming frame must only be buffered."
                    )

            self._stage_frame(
                target_np,
                source_slot=False,
            )
            if self.runner.replay_next() is None:
                raise RuntimeError(
                    "Streaming decode did not produce a pair output."
                )

            predicted_flow = self.runner.flow()[0]
            source_points = self.runner.source_points()[0]
            warped_points = self.runner.warped_points()[0]
            velocity = self.runner.velocity()[0]

        self._cached_target_timestamp_s = float(
            target_frame.timestamp_s
        )
        self._cached_target_points = target_np
        self._cached_target_indices = target_indices_np

        valid_indices = torch.from_numpy(source_indices_np).to(
            self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        return DifFlow3DEstimate(
            source_points=source_points,
            warped_points=warped_points,
            residual_flow=predicted_flow,
            velocity=velocity,
            valid_indices=valid_indices,
            source_timestamp_s=float(source_frame.timestamp_s),
            target_timestamp_s=float(target_frame.timestamp_s),
        )




class OnlineDifFlow3DBuffer:
    """Two-frame online buffer with the same push() behavior as before."""

    def __init__(
        self,
        estimator: DifFlow3DInference,
        *,
        max_frame_gap_s: float | None,
    ) -> None:
        self.estimator = estimator
        self.required_frames = estimator.required_frames
        self.max_frame_gap_s = max_frame_gap_s
        self._previous: PointCloudFrame | None = None

    @property
    def frame_count(self) -> int:
        return 0 if self._previous is None else 1

    def push(
        self,
        frame: PointCloudFrame,
    ) -> DifFlow3DEstimate | None:
        if self._previous is None:
            self._previous = frame
            return None

        dt_s = float(frame.timestamp_s - self._previous.timestamp_s)
        if dt_s <= 0.0:
            self._previous = frame
            raise ValueError(
                f"Non-increasing frame timestamps: dt={dt_s}."
            )
        if self.max_frame_gap_s is not None and dt_s > self.max_frame_gap_s:
            self._previous = frame
            return None

        source = self._previous
        self._previous = frame
        return self.estimator.infer([source, frame])


def warmup_from_sequence(
    estimator: DifFlow3DInference,
    sequence: SceneSequence,
    *,
    repetitions: int,
    point_limit: int,
) -> None:
    del point_limit
    if repetitions < 1:
        return

    identity = torch.eye(4, dtype=torch.float32)
    frames = [
        PointCloudFrame(
            points=torch.from_numpy(
                np.ascontiguousarray(
                    sequence.points[frame_index],
                    dtype=np.float32,
                )
            ),
            pose_world_sensor=identity,
            timestamp_s=float(sequence.timestamps_s[frame_index]),
        )
        for frame_index in range(estimator.required_frames)
    ]

    for _ in range(repetitions):
        estimator.reset_sampling()
        estimator.infer(frames)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(values.mean())


class RvizSceneFlowPublisher:
    """Publish online scene-flow results without affecting inference timing."""

    def __init__(
        self,
        *,
        frame_id: str,
        max_arrows: int,
        vector_scale: float,
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
                "RViz publishing requires ROS 2 Python packages: "
                "rclpy, sensor_msgs_py, geometry_msgs, and visualization_msgs."
            ) from error

        if max_arrows < 1:
            raise ValueError("--rviz-max-arrows must be positive.")
        if vector_scale <= 0.0:
            raise ValueError("--rviz-vector-scale must be positive.")

        self._rclpy = rclpy
        self._Point = Point
        self._PointCloud2 = PointCloud2
        self._point_cloud2 = point_cloud2
        self._Header = Header
        self._Marker = Marker
        self._MarkerArray = MarkerArray

        self.frame_id = frame_id
        self.max_arrows = int(max_arrows)
        self.vector_scale = float(vector_scale)

        if not rclpy.ok():
            rclpy.init(args=[])

        self.node: Node = rclpy.create_node(
            "scene_flow_online_visualizer"
        )

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.observed_publisher = self.node.create_publisher(
            PointCloud2,
            "/scene_flow/observed_target",
            qos,
        )
        self.source_publisher = self.node.create_publisher(
            PointCloud2,
            "/scene_flow/source_valid",
            qos,
        )
        self.warped_publisher = self.node.create_publisher(
            PointCloud2,
            "/scene_flow/predicted_warped",
            qos,
        )
        self.vector_publisher = self.node.create_publisher(
            MarkerArray,
            "/scene_flow/vectors",
            qos,
        )

        self.node.get_logger().info(
            "RViz publishers ready: "
            "/scene_flow/observed_target, "
            "/scene_flow/source_valid, "
            "/scene_flow/predicted_warped, "
            "/scene_flow/vectors"
        )

    def _header(self):
        header = self._Header()
        header.frame_id = self.frame_id
        header.stamp = self.node.get_clock().now().to_msg()
        return header

    def _cloud_message(self, points: np.ndarray):
        xyz = np.asarray(points, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"RViz point cloud must have shape [N,3], got {xyz.shape}."
            )
        return self._point_cloud2.create_cloud_xyz32(
            self._header(),
            np.ascontiguousarray(xyz),
        )

    def _arrow_indices(self, count: int) -> np.ndarray:
        if count <= self.max_arrows:
            return np.arange(count, dtype=np.int64)
        return np.linspace(
            0,
            count - 1,
            self.max_arrows,
            dtype=np.int64,
        )

    def _flow_marker(
        self,
        *,
        marker_id: int,
        namespace: str,
        source_points: np.ndarray,
        flow: np.ndarray,
        red: float,
        green: float,
        blue: float,
    ):
        marker = self._Marker()
        marker.header = self._header()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = self._Marker.LINE_LIST
        marker.action = self._Marker.ADD

        # LINE_LIST uses scale.x as line width.
        marker.scale.x = 0.004
        marker.color.r = float(red)
        marker.color.g = float(green)
        marker.color.b = float(blue)
        marker.color.a = 0.90
        marker.pose.orientation.w = 1.0

        source = np.asarray(source_points, dtype=np.float32)
        vector = np.asarray(flow, dtype=np.float32)
        indices = self._arrow_indices(source.shape[0])

        for index in indices:
            start = source[index]
            end = start + self.vector_scale * vector[index]

            point_start = self._Point()
            point_start.x = float(start[0])
            point_start.y = float(start[1])
            point_start.z = float(start[2])

            point_end = self._Point()
            point_end.x = float(end[0])
            point_end.y = float(end[1])
            point_end.z = float(end[2])

            marker.points.append(point_start)
            marker.points.append(point_end)

        return marker

    def publish_observed_only(self, observed_target: np.ndarray) -> None:
        self.observed_publisher.publish(
            self._cloud_message(observed_target)
        )
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_estimate(
        self,
        *,
        observed_target: np.ndarray,
        source_points: np.ndarray,
        warped_points: np.ndarray,
        predicted_flow: np.ndarray,
        ground_truth_flow: np.ndarray,
    ) -> None:
        self.observed_publisher.publish(
            self._cloud_message(observed_target)
        )
        self.source_publisher.publish(
            self._cloud_message(source_points)
        )
        self.warped_publisher.publish(
            self._cloud_message(warped_points)
        )

        markers = self._MarkerArray()
        markers.markers.append(
            self._flow_marker(
                marker_id=0,
                namespace="predicted_flow",
                source_points=source_points,
                flow=predicted_flow,
                red=1.0,
                green=0.15,
                blue=0.15,
            )
        )
        markers.markers.append(
            self._flow_marker(
                marker_id=1,
                namespace="ground_truth_flow",
                source_points=source_points,
                flow=ground_truth_flow,
                red=0.15,
                green=1.0,
                blue=0.20,
            )
        )
        self.vector_publisher.publish(markers)
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def hold(self, seconds: float) -> None:
        if seconds == 0.0:
            return

        if seconds < 0.0:
            self.node.get_logger().info(
                "Keeping RViz publishers alive until Ctrl-C."
            )
            try:
                while self._rclpy.ok():
                    self._rclpy.spin_once(
                        self.node,
                        timeout_sec=0.1,
                    )
            except KeyboardInterrupt:
                pass
            return

        end_time = time.perf_counter() + seconds
        while self._rclpy.ok() and time.perf_counter() < end_time:
            self._rclpy.spin_once(
                self.node,
                timeout_sec=0.05,
            )

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate official DifFlow3D inference on the four-superquadric "
            "dynamic safety-filter scene."
        )
    )
    parser.add_argument(
        "--difflow-repo",
        type=Path,
        default=Path.cwd(),
        help=(
            "Path to the cloned IRMVLab/DifFlow3D repository. "
            "Default: current working directory."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Default: <difflow-repo>/pretrain_weights/"
            "model_difflow_355_0.0114.pth"
        ),
    )
    parser.add_argument(
        "--model-module",
        default="model_difflow",
        help="Model module inside --difflow-repo.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 for stricter FP32 comparison.",
    )
    parser.add_argument(
        "--cuda-graph-warmup",
        type=int,
        default=10,
        help="Warmup forwards before CUDA Graph capture.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--difflow-num-points",
        type=int,
        default=1024,
        help="Fixed input size for the minimal streaming runner.",
    )
    parser.add_argument(
        "--difflow-iters",
        type=int,
        default=2,
        help=(
            "Recurrent iterations. Any positive integer is supported; "
            "the selected value is fixed into the captured CUDA Graph."
        ),
    )
    parser.add_argument(
        "--difflow-uncertainty",
        type=float,
        default=0.2,
        help="Diffusion uncertainty setting used by inference.",
    )
    parser.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help="Allow missing or unexpected checkpoint keys.",
    )
    parser.add_argument(
        "--keep-bn-running-stats",
        action="store_true",
        help=(
            "Do not reproduce evaluate.py's behavior of setting "
            "BatchNorm track_running_stats=False."
        ),
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--sensor-hz", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--warmup-point-limit",
        type=int,
        default=8192,
        help=(
            "Retained for compatibility; DifFlow3D warmup always follows "
            "--difflow-num-points."
        ),
    )
    parser.add_argument("--mesh-resolution", type=int, default=80)
    parser.add_argument(
        "--point-multiplier",
        type=int,
        default=1,
        help=(
            "Multiply each object's configured points_per_frame. "
            "The model still samples --difflow-num-points per frame."
        ),
    )
    parser.add_argument(
        "--sensor-noise-std",
        type=float,
        default=DEFAULT_SENSOR_NOISE_STD_M,
    )
    parser.add_argument(
        "--same-samples-across-frames",
        action="store_true",
        help=(
            "Reuse each object's sampled local surface points in every frame. "
            "Default: independently resample each target cloud."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--last-frame-npz",
        type=Path,
        default=None,
        help=(
            "Optionally save the final sampled source points, predicted flow, "
            "ground-truth flow, and source indices."
        ),
    )
    parser.add_argument(
        "--rviz",
        action="store_true",
        help="Publish observed clouds and flow vectors for RViz2.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help=(
            "Pace input frames according to --sensor-hz. "
            "Inference timing excludes this sleep."
        ),
    )
    parser.add_argument("--rviz-frame-id", default="world")
    parser.add_argument(
        "--rviz-max-arrows",
        type=int,
        default=512,
        help="Maximum predicted and ground-truth vectors shown per frame.",
    )
    parser.add_argument(
        "--rviz-vector-scale",
        type=float,
        default=10.0,
        help="Visual-only multiplier applied to flow vectors.",
    )
    parser.add_argument(
        "--rviz-hold-seconds",
        type=float,
        default=5.0,
        help=(
            "Keep publishers alive after the benchmark. "
            "Use a negative value to wait until Ctrl-C."
        ),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.sensor_hz <= 0.0:
        raise ValueError("--sensor-hz must be positive.")

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

    sequence = generate_sequence(
        frame_count=args.frames,
        dt_s=dt_s,
        mesh_resolution=args.mesh_resolution,
        point_multiplier=args.point_multiplier,
        sensor_noise_std_m=args.sensor_noise_std,
        same_samples_across_frames=args.same_samples_across_frames,
        seed=args.seed,
    )

    config = DifFlow3DConfig(
        repo_path=repo_path,
        checkpoint_path=checkpoint,
        model_module=args.model_module,
        enable_tf32=not args.disable_tf32,
        cuda_graph_warmup=args.cuda_graph_warmup,
        device=args.device,
        num_points=args.difflow_num_points,
        iters=args.difflow_iters,
        uncertainty=args.difflow_uncertainty,
        strict_checkpoint=not args.non_strict_checkpoint,
        disable_bn_running_stats=not args.keep_bn_running_stats,
        seed=args.seed,
        frame_dt_s=dt_s,
        max_frame_gap_s=2.0 * dt_s,
    )

    device = torch.device(config.device)
    if device.type == "cuda":
        index = cuda_index(device)
        torch.cuda.set_device(index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(index)

    total_points = sequence.points.shape[1]

    print("=" * 80)
    print("Four-superquadric DifFlow3D online test")
    print("=" * 80)
    print("Model:                 DifFlow3D (no-occlusion checkpoint)")
    print(f"Model module:          {config.model_module}")
    print("Execution backend:     streaming-cuda-graph")
    print(f"TF32 enabled:          {config.enable_tf32}")
    print(f"CUDA Graph warmup:     {config.cuda_graph_warmup}")
    print(f"Repository:            {repo_path}")
    print(f"Checkpoint:            {checkpoint}")
    print(f"Frames:                {args.frames}")
    print(f"Scene points/frame:    {total_points}")
    print(f"Model points/frame:    {config.num_points}")
    print(f"Point multiplier:      {args.point_multiplier}")
    print(f"DifFlow iterations:    {config.iters}")
    print(f"Uncertainty setting:   {config.uncertainty:.6f}")
    print(f"Sensor rate:           {args.sensor_hz:.3f} Hz")
    print(f"Sensor period:         {period_ms:.3f} ms")
    print(
        "Noise sigma:          "
        f"{args.sensor_noise_std:.6f} m per coordinate"
    )
    print(
        "Surface resampling:   "
        f"{'fixed samples' if args.same_samples_across_frames else 'independent each frame'}"
    )
    print(
        "Model frame sampling:  "
        "independent per frame, reused across adjacent pairs"
    )
    print("")
    print("Objects:")
    for spec in OBSTACLE_SPECS:
        print(
            f"  {spec.name:24s} "
            f"mode={spec.motion_mode:10s} "
            f"base_points={spec.points_per_frame}"
        )

    synchronize(device)
    load_start = time.perf_counter()
    estimator = DifFlow3DInference(config)
    synchronize(device)
    load_ms = 1000.0 * (time.perf_counter() - load_start)

    exact_checkpoint = estimator.checkpoint_report.is_exact

    print("")
    print(f"Model load time:       {load_ms:.1f} ms")
    print(f"Required frames:       {estimator.required_frames}")
    print(f"Exact checkpoint:      {exact_checkpoint}")
    if not exact_checkpoint:
        print(
            "Missing checkpoint keys: "
            f"{estimator.checkpoint_report.missing_keys}"
        )
        print(
            "Unexpected checkpoint keys: "
            f"{estimator.checkpoint_report.unexpected_keys}"
        )

    if args.warmup > 0:
        print(
            f"Warmup:                {args.warmup} forward(s), "
            f"{config.num_points} sampled points/frame"
        )
        warmup_from_sequence(
            estimator,
            sequence,
            repetitions=args.warmup,
            point_limit=args.warmup_point_limit,
        )
        synchronize(device)
        estimator.reset_sampling()

    online = OnlineDifFlow3DBuffer(
        estimator,
        max_frame_gap_s=config.max_frame_gap_s,
    )

    rviz_publisher = (
        RvizSceneFlowPublisher(
            frame_id=args.rviz_frame_id,
            max_arrows=args.rviz_max_arrows,
            vector_scale=args.rviz_vector_scale,
        )
        if args.rviz
        else None
    )

    identity = torch.eye(4, dtype=torch.float32)

    latencies_ms: list[float] = []
    all_epe: list[np.ndarray] = []
    # all_velocity_epe: list[np.ndarray] = []
    all_average_velocity_epe: list[np.ndarray] = []
    all_target_velocity_epe: list[np.ndarray] = []
    all_dynamic: list[np.ndarray] = []
    valid_ratios: list[float] = []
    per_object_epe: dict[str, list[np.ndarray]] = {
        name: []
        for name in sequence.object_names
    }
    per_frame: list[dict[str, object]] = []

    last_payload: dict[str, np.ndarray] | None = None

    print("")
    print("Streaming frames")
    print("-" * 80)

    stream_wall_start = time.perf_counter()

    for target_index in range(args.frames):
        if args.realtime:
            release_time = stream_wall_start + target_index * dt_s
            sleep_s = release_time - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        frame = PointCloudFrame(
            points=torch.from_numpy(sequence.points[target_index]),
            pose_world_sensor=identity,
            timestamp_s=float(sequence.timestamps_s[target_index]),
        )

        synchronize(device)
        start = time.perf_counter()
        estimate = online.push(frame)
        synchronize(device)
        latency_ms = 1000.0 * (time.perf_counter() - start)

        if estimate is None:
            if rviz_publisher is not None:
                rviz_publisher.publish_observed_only(
                    sequence.points[target_index]
                )

            print(
                f"frame {target_index:03d}: buffering "
                f"{online.frame_count}/{online.required_frames}"
            )
            continue

        # All retained models return P_(k-1) -> P_k.
        source_index = target_index - 1

        expected_source_timestamp = sequence.timestamps_s[source_index]
        expected_target_timestamp = sequence.timestamps_s[target_index]
        if not np.isclose(
            estimate.source_timestamp_s,
            expected_source_timestamp,
        ):
            raise RuntimeError(
                "Unexpected source timestamp: "
                f"{estimate.source_timestamp_s} vs "
                f"{expected_source_timestamp}."
            )
        if not np.isclose(
            estimate.target_timestamp_s,
            expected_target_timestamp,
        ):
            raise RuntimeError(
                "Unexpected target timestamp: "
                f"{estimate.target_timestamp_s} vs "
                f"{expected_target_timestamp}."
            )

        valid_indices = (
            estimate.valid_indices.detach().cpu().numpy().astype(np.int64)
        )
        predicted_flow = estimate.residual_flow.detach().cpu().numpy()
        predicted_velocity = estimate.velocity.detach().cpu().numpy()

        # gt_flow = sequence.gt_flow[source_index, valid_indices]
        # gt_velocity = gt_flow / np.float32(dt_s)

        # epe = np.linalg.norm(predicted_flow - gt_flow, axis=1)
        # velocity_epe = np.linalg.norm(
        #     predicted_velocity - gt_velocity,
        #     axis=1,
        # )

        gt_flow = sequence.gt_flow[
            source_index,
            valid_indices,
        ]

        # Interval-average ground-truth velocity.
        gt_velocity_average = (
            gt_flow / np.float32(dt_s)
        )

        # Analytic instantaneous velocity at the target timestamp.
        gt_velocity_target = (
            sequence.gt_velocity_target[
                source_index,
                valid_indices,
            ]
        )

        epe = np.linalg.norm(
            predicted_flow - gt_flow,
            axis=1,
        )

        average_velocity_epe = np.linalg.norm(
            predicted_velocity
            - gt_velocity_average,
            axis=1,
        )

        target_velocity_epe = np.linalg.norm(
            predicted_velocity
            - gt_velocity_target,
            axis=1,
        )

        object_ids = sequence.object_ids[valid_indices]
        dynamic = sequence.dynamic_mask[valid_indices]

        latencies_ms.append(latency_ms)
        all_epe.append(epe)
        # all_velocity_epe.append(velocity_epe)
        all_average_velocity_epe.append(average_velocity_epe)
        all_target_velocity_epe.append(target_velocity_epe)
        all_dynamic.append(dynamic)
        valid_ratios.append(valid_indices.size / total_points)

        object_frame_metrics: dict[str, float | int | None] = {}
        for object_id, object_name in enumerate(sequence.object_names):
            mask = object_ids == object_id
            object_errors = epe[mask]
            per_object_epe[object_name].append(object_errors)

            object_frame_metrics[object_name] = (
                float(object_errors.mean())
                if object_errors.size
                else None
            )
            object_frame_metrics[f"{object_name}_valid_points"] = int(
                mask.sum()
            )

        moving_gt_norm = np.linalg.norm(gt_flow[dynamic], axis=1)
        moving_pred_norm = np.linalg.norm(
            predicted_flow[dynamic],
            axis=1,
        )
        direction_mask = (
            dynamic
            & (np.linalg.norm(gt_flow, axis=1) > 1.0e-8)
            & (np.linalg.norm(predicted_flow, axis=1) > 1.0e-8)
        )
        if np.any(direction_mask):
            cosine = np.sum(
                predicted_flow[direction_mask]
                * gt_flow[direction_mask],
                axis=1,
            ) / (
                np.linalg.norm(
                    predicted_flow[direction_mask],
                    axis=1,
                )
                * np.linalg.norm(
                    gt_flow[direction_mask],
                    axis=1,
                )
            )
            mean_direction_cosine = float(np.clip(cosine, -1.0, 1.0).mean())
        else:
            mean_direction_cosine = None

        dynamic_epe = safe_mean(epe[dynamic])
        static_epe = safe_mean(epe[~dynamic])

        row: dict[str, object] = {
            "source_index": source_index,
            "target_index": target_index,
            "source_timestamp_s": float(expected_source_timestamp),
            "target_timestamp_s": float(expected_target_timestamp),
            "latency_ms": latency_ms,
            "valid_points": int(valid_indices.size),
            "valid_ratio": float(valid_indices.size / total_points),
            "mean_epe_m": float(epe.mean()),
            "p95_epe_m": percentile(epe, 95.0),
            # "mean_velocity_epe_mps": float(velocity_epe.mean()),
            "mean_average_velocity_epe_mps": float(average_velocity_epe.mean()),
            "mean_target_velocity_epe_mps": float(target_velocity_epe.mean()),
            "dynamic_epe_m": dynamic_epe,
            "static_epe_m": static_epe,
            "mean_moving_direction_cosine": mean_direction_cosine,
            "mean_gt_moving_flow_m": (
                float(moving_gt_norm.mean())
                if moving_gt_norm.size
                else None
            ),
            "mean_predicted_moving_flow_m": (
                float(moving_pred_norm.mean())
                if moving_pred_norm.size
                else None
            ),
            "objects": object_frame_metrics,
        }
        per_frame.append(row)

        print(
            f"{source_index:03d}->{target_index:03d} | "
            f"{latency_ms:8.2f} ms | "
            f"flow EPE {epe.mean():8.5f} m | "
            f"avg-vel EPE "
            f"{average_velocity_epe.mean():8.4f} m/s | "
            f"target-vel EPE "
            f"{target_velocity_epe.mean():8.4f} m/s"
        )

        source_points_cpu = (
            estimate.source_points.detach().cpu().numpy()
        )
        warped_points_cpu = (
            estimate.warped_points.detach().cpu().numpy()
        )

        last_payload = {
            "source_points": source_points_cpu,
            "warped_points": warped_points_cpu,
            "predicted_flow": predicted_flow,
            "predicted_velocity": predicted_velocity,
            "gt_flow": gt_flow,
            # "gt_velocity": gt_velocity,
            "gt_velocity_average": gt_velocity_average,
            "gt_velocity_target": gt_velocity_target,
            "valid_indices": valid_indices,
            "object_ids": object_ids,
            "dynamic_mask": dynamic,
        }

        if rviz_publisher is not None:
            rviz_publisher.publish_estimate(
                observed_target=sequence.points[target_index],
                source_points=source_points_cpu,
                warped_points=warped_points_cpu,
                predicted_flow=predicted_flow,
                ground_truth_flow=gt_flow,
            )

    if not latencies_ms:
        raise RuntimeError("No scene-flow estimate was produced.")

    latency = np.asarray(latencies_ms, dtype=np.float64)
    epe = np.concatenate(all_epe)
    # velocity_epe = np.concatenate(all_velocity_epe)
    average_velocity_epe = np.concatenate(all_average_velocity_epe)
    target_velocity_epe = np.concatenate(all_target_velocity_epe)
    dynamic = np.concatenate(all_dynamic)

    per_object_summary: dict[str, dict[str, float | int | str]] = {}
    for object_id, (spec, object_name) in enumerate(
        zip(OBSTACLE_SPECS, sequence.object_names)
    ):
        chunks = [
            values
            for values in per_object_epe[object_name]
            if values.size
        ]
        if chunks:
            object_errors = np.concatenate(chunks)
            per_object_summary[object_name] = {
                "motion_mode": spec.motion_mode,
                "samples": int(object_errors.size),
                "mean_epe_m": float(object_errors.mean()),
                "median_epe_m": float(np.median(object_errors)),
                "p95_epe_m": percentile(object_errors, 95.0),
                "max_gt_flow_m": float(
                    sequence.max_gt_flow_by_object_m[object_name]
                ),
            }

    median_latency = float(np.median(latency))
    p95_latency = percentile(latency, 95.0)

    result: dict[str, object] = {
        "scene": {
            "name": "four_superquadric_dynamic_scene",
            "specs": [asdict(spec) for spec in OBSTACLE_SPECS],
            "frames": args.frames,
            "dt_s": dt_s,
            "sensor_hz": args.sensor_hz,
            "points_per_frame": total_points,
            "point_multiplier": args.point_multiplier,
            "mesh_resolution": args.mesh_resolution,
            "sensor_noise_std_m": args.sensor_noise_std,
            "same_samples_across_frames": (
                args.same_samples_across_frames
            ),
            "seed": args.seed,
        },
        "model": {
            "name": "difflow3d",
            "variant": "minimal_streaming_cuda_graph",
            "model_module": config.model_module,
            "execution_backend": "streaming-cuda-graph",
            "tf32_enabled": config.enable_tf32,
            "cuda_graph_warmup": config.cuda_graph_warmup,
            "repository": str(repo_path),
            "checkpoint": str(checkpoint),
            "exact_checkpoint": exact_checkpoint,
            "required_frames": estimator.required_frames,
            "num_points": config.num_points,
            "iters": config.iters,
            "uncertainty": config.uncertainty,
            "load_time_ms": load_ms,
        },
        "accuracy": {
            "mean_epe_m": float(epe.mean()),
            "median_epe_m": float(np.median(epe)),
            "p95_epe_m": percentile(epe, 95.0),
            "mean_average_velocity_epe_mps": float(
                average_velocity_epe.mean()
            ),
            "median_average_velocity_epe_mps": float(
                np.median(
                    average_velocity_epe
                )
            ),
            "p95_average_velocity_epe_mps": percentile(
                average_velocity_epe,
                95.0,
            ),

            "mean_target_velocity_epe_mps": float(
                target_velocity_epe.mean()
            ),
            "median_target_velocity_epe_mps": float(
                np.median(
                    target_velocity_epe
                )
            ),
            "p95_target_velocity_epe_mps": percentile(
                target_velocity_epe,
                95.0,
            ),
            "dynamic_mean_epe_m": safe_mean(epe[dynamic]),
            "static_mean_epe_m": safe_mean(epe[~dynamic]),
            "per_object": per_object_summary,
        },
        "efficiency": {
            "sensor_period_ms": period_ms,
            "mean_latency_ms": float(latency.mean()),
            "median_latency_ms": median_latency,
            "p90_latency_ms": percentile(latency, 90.0),
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": percentile(latency, 99.0),
            "max_latency_ms": float(latency.max()),
            "deadline_miss_ratio": float(
                np.mean(latency > period_ms)
            ),
            "median_realtime_factor": median_latency / period_ms,
            "p95_realtime_factor": p95_latency / period_ms,
            "sustainable_hz_from_median": 1000.0 / median_latency,
            "mean_valid_ratio": float(np.mean(valid_ratios)),
        },
        "per_frame": per_frame,
    }

    if device.type == "cuda":
        index = cuda_index(device)
        result["gpu_memory_mib"] = {
            "peak_allocated": float(
                torch.cuda.max_memory_allocated(index) / 1024**2
            ),
            "reserved": float(
                torch.cuda.memory_reserved(index) / 1024**2
            ),
        }

    accuracy = result["accuracy"]
    efficiency = result["efficiency"]

    print("")
    print("=" * 80)
    print("Aggregate accuracy")
    print("=" * 80)
    print(f"Mean EPE:              {accuracy['mean_epe_m']:.6f} m")
    print(f"Median EPE:            {accuracy['median_epe_m']:.6f} m")
    print(f"P95 EPE:               {accuracy['p95_epe_m']:.6f} m")
    print(
        "Mean avg-vel EPE:     "
        f"{accuracy['mean_average_velocity_epe_mps']:.6f} m/s"
    )

    print(
        "Median avg-vel EPE:   "
        f"{accuracy['median_average_velocity_epe_mps']:.6f} m/s"
    )

    print(
        "P95 avg-vel EPE:      "
        f"{accuracy['p95_average_velocity_epe_mps']:.6f} m/s"
    )

    print(
        "Mean target-vel EPE:  "
        f"{accuracy['mean_target_velocity_epe_mps']:.6f} m/s"
    )

    print(
        "Median target-vel EPE:"
        f" {accuracy['median_target_velocity_epe_mps']:.6f} m/s"
    )

    print(
        "P95 target-vel EPE:   "
        f"{accuracy['p95_target_velocity_epe_mps']:.6f} m/s"
    )
    print(
        "Static mean EPE:      "
        f"{accuracy['static_mean_epe_m']:.6f} m"
    )
    print(
        "Dynamic mean EPE:     "
        f"{accuracy['dynamic_mean_epe_m']:.6f} m"
    )

    print("")
    print("Per-object EPE")
    print("-" * 80)
    for object_name, metrics in per_object_summary.items():
        print(
            f"{object_name:24s} "
            f"mode={metrics['motion_mode']:10s} "
            f"mean={metrics['mean_epe_m']:.6f} m "
            f"p95={metrics['p95_epe_m']:.6f} m "
            f"max_gt={metrics['max_gt_flow_m']:.6f} m"
        )

    print("")
    print("=" * 80)
    print("Online efficiency")
    print("=" * 80)
    print(f"Sensor period:         {period_ms:.3f} ms")
    print(f"Mean latency:          {efficiency['mean_latency_ms']:.3f} ms")
    print(f"Median latency:        {efficiency['median_latency_ms']:.3f} ms")
    print(f"P90 latency:           {efficiency['p90_latency_ms']:.3f} ms")
    print(f"P95 latency:           {efficiency['p95_latency_ms']:.3f} ms")
    print(f"Maximum latency:       {efficiency['max_latency_ms']:.3f} ms")
    print(
        "Deadline misses:      "
        f"{100.0 * efficiency['deadline_miss_ratio']:.2f}%"
    )
    print(
        "Median RT factor:     "
        f"{efficiency['median_realtime_factor']:.3f}x"
    )
    print(
        "P95 RT factor:        "
        f"{efficiency['p95_realtime_factor']:.3f}x"
    )
    print(
        "Sustainable rate:     "
        f"{efficiency['sustainable_hz_from_median']:.2f} Hz"
    )
    print(
        "Mean valid ratio:     "
        f"{100.0 * efficiency['mean_valid_ratio']:.2f}%"
    )

    if "gpu_memory_mib" in result:
        memory = result["gpu_memory_mib"]
        print(
            "Peak GPU memory:     "
            f"{memory['peak_allocated']:.1f} MiB"
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
            raise RuntimeError("No final estimate is available to save.")
        args.last_frame_npz.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        np.savez_compressed(args.last_frame_npz, **last_payload)
        print(f"Saved final-frame arrays: {args.last_frame_npz}")

    if rviz_publisher is not None:
        print(
            "\nRViz topics:\n"
            "  /scene_flow/observed_target\n"
            "  /scene_flow/source_valid\n"
            "  /scene_flow/predicted_warped\n"
            "  /scene_flow/vectors"
        )
        print(
            "Predicted vectors are red; ground-truth vectors are green. "
            f"Vector display scale: {args.rviz_vector_scale:.3f}x."
        )
        rviz_publisher.hold(args.rviz_hold_seconds)
        rviz_publisher.close()


if __name__ == "__main__":
    main()
