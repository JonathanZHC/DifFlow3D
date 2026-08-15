"""Synthetic four-superquadric scene used by deployment benchmarks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.spatial.transform import Rotation

MotionMode = Literal["static", "sinusoidal", "twist"]

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
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    motion_frequency: float = 0.0
    motion_phase: float = 0.0
    twist_world: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )


def make_obstacle_specs(
    dimension_factor: float,
    motion: bool,
) -> tuple[SQSpec, ...]:
    """Create the synthetic scene with consistent metric scaling.

    Lengths, translations, translational motion amplitudes and linear twists
    scale with ``dimension_factor``.  Angles, angular motion and frequencies do
    not.
    """
    d = float(dimension_factor)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("--dimension-factor must be positive and finite.")

    return (
        SQSpec(
            name="main_box",
            axes=(0.18 * d, 0.55 * d, 0.40 * d),
            eps1=1.0,
            eps2=1.0,
            initial_translation=(0.25 * d, 0.0, 0.30 * d),
            initial_axis_angle=(0.0, 0.0, 0.0),
            points_per_frame=1024,
            motion_mode="static",
        ),
        SQSpec(
            name="left_rounded_pillar",
            axes=(0.12 * d, 0.12 * d, 0.55 * d),
            eps1=0.45,
            eps2=1.0,
            initial_translation=(0.15 * d, 0.42 * d, 0.35 * d),
            initial_axis_angle=(0.0, 0.35, 0.15),
            points_per_frame=1024,
            motion_mode="sinusoidal" if motion else "static",
            motion_amplitude=(0.0, 5 * 0.20 * d, 0.0, 0.0, 0.0, 0.0),
            motion_frequency=0.15,
            motion_phase=0.0,
        ),
        SQSpec(
            name="right_flat_ellipsoid",
            axes=(0.35 * d, 0.16 * d, 0.14 * d),
            eps1=1.0,
            eps2=1.0,
            initial_translation=(0.45 * d, -0.38 * d, 0.26 * d),
            initial_axis_angle=(0.25, 0.0, -0.55),
            points_per_frame=1024,
            motion_mode="sinusoidal" if motion else "static",
            # Rotational amplitude is angular, so it is not scaled by d.
            motion_amplitude=(0.0, 0.0, 0.0, 0.0, 0.0, 5 * 0.50),
            motion_frequency=0.20,
            motion_phase=0.0,
        ),
        SQSpec(
            name="front_tilted_box",
            axes=(0.22 * d, 0.18 * d, 0.28 * d),
            eps1=0.35,
            eps2=0.35,
            initial_translation=(0.62 * d, 0.12 * d, 0.42 * d),
            initial_axis_angle=(0.45, -0.25, 0.75),
            points_per_frame=1024,
            motion_mode="twist" if motion else "static",
            # First three twist entries are linear velocity [m/s].
            twist_world=(10 * 0.02 * d, 0.0, 0.0, 0.0, 0.0, 0.0),
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
def distribute_point_counts(
    total_points: int,
    specs: tuple[SQSpec, ...],
) -> list[int]:
    """Distribute an exact total across objects using their configured weights."""
    if total_points < len(specs):
        raise ValueError(
            f"--all-points must be at least {len(specs)}."
        )
    weights = np.asarray(
        [spec.points_per_frame for spec in specs],
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
    specs: tuple[SQSpec, ...],
) -> SceneSequence:
    if frame_count < 3:
        raise ValueError("--frames must be at least 3.")
    if dt_s <= 0.0:
        raise ValueError("Frame interval must be positive.")
    if sensor_noise_std_m < 0.0:
        raise ValueError("--sensor-noise-std must be non-negative.")

    rng = np.random.default_rng(seed)
    timestamps_s = np.arange(frame_count, dtype=np.float64) * dt_s
    counts = distribute_point_counts(total_points, specs)

    samplers: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    fixed_samples: dict[str, np.ndarray] = {}

    for spec, count in zip(specs, counts):
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
            for spec, count in zip(specs, counts)
        ]
    )
    max_gt_flow_by_object_m = {
        spec.name: 0.0 for spec in specs
    }

    for frame_index, time_s in enumerate(timestamps_s):
        point_offset = 0
        for spec, count in zip(specs, counts):
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
        object_names=tuple(spec.name for spec in specs),
        dynamic_mask=dynamic_mask,
        max_gt_flow_by_object_m=max_gt_flow_by_object_m,
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
        specs: tuple[SQSpec, ...],
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
        self.specs = tuple(specs)
        self.counts = distribute_point_counts(total_points, self.specs)

        self.samplers: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self.fixed_samples: dict[str, np.ndarray] = {}

        base_rng = np.random.default_rng(self.seed)
        for spec, count in zip(self.specs, self.counts):
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
                for spec, count in zip(self.specs, self.counts)
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
        for spec, count in zip(self.specs, self.counts):
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
