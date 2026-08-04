#!/usr/bin/env python3
"""Create two calibrated RGB-D cameras and retain CUDA sensor arrays.

The camera optical frame follows the ROS/OpenCV convention:
+x right, +y down, +z forward.

Isaac Sim RGB and depth annotators remain on CUDA. Optional RGB and depth
corruption is executed entirely by NVIDIA Warp kernels before any device-to-
host copy. The active GPU arrays are retained for full per-camera point-cloud
back-projection, while CPU copies are used only for ROS Image publication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import omni.replicator.core as rep
import warp as wp
from pxr import Gf, UsdGeom

from camera_corruption_warp import (
    depth_camera_corruption_wp,
    rgb_camera_corruption_wp,
)


MAX_DEPTH_M = 10.0
WARP_DEVICE = "cuda:0"


@dataclass(frozen=True)
class CameraSpec:
    name: str
    prim_path: str
    position_world: tuple[float, float, float]
    look_at_world: tuple[float, float, float]
    optical_frame_id: str
    focal_length_mm: float
    horizontal_aperture_mm: float = 20.955
    near_m: float = 0.05
    far_m: float = MAX_DEPTH_M


CAMERA_SPECS = (
    CameraSpec(
        name="camera_0",
        prim_path="/World/Cameras/camera_0",
        position_world=(1.55, -1.70, 1.55),
        look_at_world=(0.00, 0.00, 0.88),
        optical_frame_id="camera_0_optical_frame",
        focal_length_mm=24.0,
    ),
    CameraSpec(
        name="camera_1",
        prim_path="/World/Cameras/camera_1",
        position_world=(-1.45, -1.45, 1.35),
        look_at_world=(0.05, 0.05, 0.86),
        optical_frame_id="camera_1_optical_frame",
        focal_length_mm=28.0,
    ),
)


@dataclass(frozen=True)
class CameraRigConfig:
    width: int
    height: int
    world_frame_id: str = "world"
    max_depth_m: float = MAX_DEPTH_M
    camera_specs: tuple[CameraSpec, ...] = CAMERA_SPECS


@dataclass(frozen=True)
class CorruptionConfig:
    enabled: bool
    corrupt_rgb: bool = True
    corrupt_depth: bool = True
    seed: int = 7

    rgb_noise_std_255: float = 2.0
    exposure_fraction: float = 0.10

    depth_noise_base_m: float = 0.0015
    depth_noise_quadratic: float = 0.0015
    depth_quantization_m: float = 0.001
    random_dropout_probability: float = 0.005
    edge_dropout_probability: float = 0.10
    edge_threshold_m: float = 0.025

    def validate(self) -> None:
        if self.enabled and not (
            self.corrupt_rgb or self.corrupt_depth
        ):
            raise ValueError(
                "--corrupt requires RGB corruption, depth corruption, "
                "or both."
            )


@dataclass
class CameraRuntime:
    spec: CameraSpec
    camera_prim: Any
    render_product: Any
    rgb_annotator: Any
    depth_annotator: Any
    K: np.ndarray
    T_world_from_camera_optical: np.ndarray
    ray_x: np.ndarray
    ray_y: np.ndarray
    rgb_corrupted_gpu: Any | None = None
    depth_corrupted_gpu: Any | None = None


@dataclass
class CameraFrame:
    runtime: CameraRuntime
    stream_label: str
    rgb: np.ndarray
    depth_m: np.ndarray
    rgb_gpu: Any
    depth_gpu: Any


@dataclass
class _PendingCapture:
    runtime: CameraRuntime
    rgb_gpu: Any
    depth_gpu: Any
    active_rgb_gpu: Any
    active_depth_gpu: Any


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def rotation_matrix_to_quaternion_xyzw(
    rotation: np.ndarray,
) -> np.ndarray:
    """Convert a 3x3 rotation matrix to [x, y, z, w]."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"Rotation matrix must be 3x3, got {matrix.shape}."
        )

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif (
        matrix[0, 0] > matrix[1, 1]
        and matrix[0, 0] > matrix[2, 2]
    ):
        scale = math.sqrt(
            1.0
            + matrix[0, 0]
            - matrix[1, 1]
            - matrix[2, 2]
        ) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(
            1.0
            + matrix[1, 1]
            - matrix[0, 0]
            - matrix[2, 2]
        ) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(
            1.0
            + matrix[2, 2]
            - matrix[0, 0]
            - matrix[1, 1]
        ) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def make_camera_pose(
    position_world: tuple[float, float, float],
    look_at_world: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-from-optical transform and the USD camera quaternion."""

    eye = np.asarray(position_world, dtype=np.float64)
    target = np.asarray(look_at_world, dtype=np.float64)
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)

    z_forward = _normalize(target - eye)
    x_right = _normalize(np.cross(z_forward, world_up))
    y_down = _normalize(np.cross(z_forward, x_right))

    rotation_world_from_optical = np.column_stack(
        (x_right, y_down, z_forward)
    )

    # USD Camera local axes are +x right, +y up and -z forward.
    optical_from_usd = np.diag([1.0, -1.0, -1.0])
    rotation_world_from_usd = (
        rotation_world_from_optical @ optical_from_usd
    )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_world_from_optical
    transform[:3, 3] = eye

    quaternion_usd_xyzw = rotation_matrix_to_quaternion_xyzw(
        rotation_world_from_usd
    )
    return transform, quaternion_usd_xyzw


def camera_intrinsics(
    width: int,
    height: int,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
) -> tuple[np.ndarray, float]:
    """Return the pinhole intrinsic matrix and vertical aperture."""

    vertical_aperture_mm = (
        horizontal_aperture_mm * float(height) / float(width)
    )
    fx = float(width) * focal_length_mm / horizontal_aperture_mm
    fy = float(height) * focal_length_mm / vertical_aperture_mm
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5

    K = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, vertical_aperture_mm


def _set_usd_camera_pose(
    prim,
    translation_world: np.ndarray,
    quaternion_usd_xyzw: np.ndarray,
) -> None:
    x, y, z, w = [
        float(value) for value in quaternion_usd_xyzw
    ]

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(
        Gf.Vec3d(
            *[float(value) for value in translation_world]
        )
    )
    xformable.AddOrientOp().Set(
        Gf.Quatf(w, Gf.Vec3f(x, y, z))
    )


def _precompute_rays(
    width: int,
    height: int,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    ray_x = (u - float(K[0, 2])) / float(K[0, 0])
    ray_y = (v - float(K[1, 2])) / float(K[1, 1])
    return ray_x, ray_y


def create_cameras(
    stage,
    rig: CameraRigConfig,
    corruption: CorruptionConfig,
) -> list[CameraRuntime]:
    """Create cameras and attach plain CUDA annotators.

    Corruption is launched manually with Warp after get_data(). This avoids
    relying on an augmented Replicator depth graph and keeps the active stream
    explicit.
    """

    if rig.width <= 0 or rig.height <= 0:
        raise ValueError("Camera resolution must be positive.")

    corruption.validate()
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError(
            "NVIDIA Warp cannot access CUDA. Start Docker with --gpus all."
        )

    cameras: list[CameraRuntime] = []

    for spec in rig.camera_specs:
        transform, quaternion_usd_xyzw = make_camera_pose(
            spec.position_world,
            spec.look_at_world,
        )

        camera = UsdGeom.Camera.Define(stage, spec.prim_path)
        camera.CreateFocalLengthAttr(float(spec.focal_length_mm))
        camera.CreateHorizontalApertureAttr(
            float(spec.horizontal_aperture_mm)
        )

        K, vertical_aperture_mm = camera_intrinsics(
            rig.width,
            rig.height,
            spec.focal_length_mm,
            spec.horizontal_aperture_mm,
        )
        camera.CreateVerticalApertureAttr(
            float(vertical_aperture_mm)
        )
        camera.CreateHorizontalApertureOffsetAttr(0.0)
        camera.CreateVerticalApertureOffsetAttr(0.0)
        camera.CreateClippingRangeAttr(
            Gf.Vec2f(float(spec.near_m), float(spec.far_m))
        )

        _set_usd_camera_pose(
            camera.GetPrim(),
            np.asarray(spec.position_world, dtype=np.float64),
            quaternion_usd_xyzw,
        )

        render_product = rep.create.render_product(
            spec.prim_path,
            resolution=(rig.width, rig.height),
            name=f"{spec.name}_render_product",
        )

        rgb_annotator = rep.annotators.get(
            "rgb",
            device="cuda",
        )
        depth_annotator = rep.annotators.get(
            "distance_to_image_plane",
            device="cuda",
        )
        rgb_annotator.attach(render_product)
        depth_annotator.attach(render_product)

        ray_x, ray_y = _precompute_rays(
            rig.width,
            rig.height,
            K,
        )

        cameras.append(
            CameraRuntime(
                spec=spec,
                camera_prim=camera,
                render_product=render_product,
                rgb_annotator=rgb_annotator,
                depth_annotator=depth_annotator,
                K=K,
                T_world_from_camera_optical=transform,
                ray_x=ray_x,
                ray_y=ray_y,
            )
        )

    return cameras


def _unwrap_annotator_data(data: Any, label: str):
    if isinstance(data, dict):
        if "data" not in data:
            raise RuntimeError(
                f"{label} annotator returned a dictionary without 'data'."
            )
        data = data["data"]

    if isinstance(data, wp.array):
        return data

    array = np.asarray(data)
    if array.size == 0:
        raise RuntimeError(f"{label} annotator returned empty data.")
    return wp.array(
        array,
        dtype=wp.uint8 if array.dtype == np.uint8 else wp.float32,
        device=WARP_DEVICE,
    )


def _ensure_rgb_output(
    runtime: CameraRuntime,
    source,
):
    if (
        runtime.rgb_corrupted_gpu is None
        or tuple(runtime.rgb_corrupted_gpu.shape)
        != tuple(source.shape)
    ):
        runtime.rgb_corrupted_gpu = wp.empty(
            source.shape,
            dtype=wp.uint8,
            device=WARP_DEVICE,
        )
    return runtime.rgb_corrupted_gpu


def _ensure_depth_output(
    runtime: CameraRuntime,
    source,
):
    if (
        runtime.depth_corrupted_gpu is None
        or tuple(runtime.depth_corrupted_gpu.shape)
        != tuple(source.shape)
    ):
        runtime.depth_corrupted_gpu = wp.empty(
            source.shape,
            dtype=wp.float32,
            device=WARP_DEVICE,
        )
    return runtime.depth_corrupted_gpu


def capture_all_cameras(
    cameras: Iterable[CameraRuntime],
    rig: CameraRigConfig,
    corruption: CorruptionConfig,
    frame_index: int,
) -> dict[str, CameraFrame]:
    """Capture one active stream per camera.

    Clean mode reads the CUDA annotators directly.
    Corruption mode launches Warp kernels directly on those CUDA arrays.
    """

    pending: list[_PendingCapture] = []

    for camera_index, runtime in enumerate(cameras):
        rgb_gpu = _unwrap_annotator_data(
            runtime.rgb_annotator.get_data(),
            f"{runtime.spec.name}/rgb",
        )
        depth_gpu = _unwrap_annotator_data(
            runtime.depth_annotator.get_data(),
            f"{runtime.spec.name}/depth",
        )

        active_rgb_gpu = rgb_gpu
        active_depth_gpu = depth_gpu

        frame_seed = (
            corruption.seed
            + frame_index * 1_000_003
            + camera_index * 100_003
        )

        if corruption.enabled and corruption.corrupt_rgb:
            active_rgb_gpu = _ensure_rgb_output(
                runtime,
                rgb_gpu,
            )
            wp.launch(
                kernel=rgb_camera_corruption_wp,
                dim=(rig.height, rig.width),
                inputs=[
                    rgb_gpu,
                    active_rgb_gpu,
                    corruption.rgb_noise_std_255,
                    corruption.exposure_fraction,
                    frame_seed + 11,
                ],
                device=WARP_DEVICE,
            )

        if corruption.enabled and corruption.corrupt_depth:
            if depth_gpu.ndim == 1:
                expected_size = rig.height * rig.width

                if depth_gpu.size != expected_size:
                    raise RuntimeError(
                        f"{runtime.spec.name} flattened depth has "
                        f"{depth_gpu.size} elements, expected "
                        f"{rig.height}x{rig.width}={expected_size}."
                    )

                depth_gpu = depth_gpu.reshape(
                    (rig.height, rig.width)
                )

            active_depth_gpu = _ensure_depth_output(
                runtime,
                depth_gpu,
            )

            wp.launch(
                kernel=depth_camera_corruption_wp,
                dim=(rig.height, rig.width),
                inputs=[
                    depth_gpu,
                    active_depth_gpu,
                    corruption.depth_noise_base_m,
                    corruption.depth_noise_quadratic,
                    corruption.depth_quantization_m,
                    corruption.random_dropout_probability,
                    corruption.edge_dropout_probability,
                    corruption.edge_threshold_m,
                    frame_seed + 29,
                ],
                device=WARP_DEVICE,
            )

        pending.append(
            _PendingCapture(
                runtime=runtime,
                rgb_gpu=rgb_gpu,
                depth_gpu=depth_gpu,
                active_rgb_gpu=active_rgb_gpu,
                active_depth_gpu=active_depth_gpu,
            )
        )

    if corruption.enabled:
        wp.synchronize()

    frames: dict[str, CameraFrame] = {}
    stream_label = (
        "corrupted" if corruption.enabled else "clean"
    )

    for item in pending:
        rgb_raw = item.active_rgb_gpu.numpy()
        depth_raw = item.active_depth_gpu.numpy()

        if rgb_raw.ndim != 3 or rgb_raw.shape[2] < 3:
            raise RuntimeError(
                f"{item.runtime.spec.name} RGB shape is {rgb_raw.shape}."
            )

        rgb = np.ascontiguousarray(
            rgb_raw[..., :3],
            dtype=np.uint8,
        )
        depth = np.asarray(
            np.squeeze(depth_raw),
            dtype=np.float32,
        )
        if depth.ndim != 2:
            raise RuntimeError(
                f"{item.runtime.spec.name} depth shape is "
                f"{depth_raw.shape}."
            )

        frames[item.runtime.spec.name] = CameraFrame(
            runtime=item.runtime,
            stream_label=stream_label,
            rgb=rgb,
            depth_m=depth,
            rgb_gpu=item.active_rgb_gpu,
            depth_gpu=item.active_depth_gpu,
        )

    return frames
