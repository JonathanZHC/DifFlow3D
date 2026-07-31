#!/usr/bin/env python3
"""两台标定 RGB-D 相机、噪声模型与点云生成。

本模块不创建 SimulationApp，也不创建 ROS 节点。
相机 optical frame 使用 ROS/OpenCV 约定：
    +x 向右，+y 向下，+z 向前。
世界坐标使用 Isaac Sim 约定：
    +z 向上，单位米。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import omni.replicator.core as rep
from PIL import Image
from pxr import Gf, UsdGeom


@dataclass(frozen=True)
class CameraSpec:
    name: str
    prim_path: str
    position_world: tuple[float, float, float]
    look_at_world: tuple[float, float, float]
    optical_frame_id: str
    focal_length_mm: float = 24.0
    horizontal_aperture_mm: float = 20.955
    near_m: float = 0.05
    far_m: float = 10.0


DEFAULT_CAMERA_SPECS = (
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
    width: int = 640
    height: int = 480
    max_depth_m: float = 5.0
    point_stride: int = 1
    world_frame_id: str = "world"
    camera_specs: tuple[CameraSpec, ...] = DEFAULT_CAMERA_SPECS


@dataclass(frozen=True)
class CorruptionConfig:
    enabled: bool = False
    seed: int = 7

    # RGB。
    rgb_noise_std_255: float = 2.0
    exposure_fraction: float = 0.10

    # 深度噪声 sigma(z)=base + quadratic*z^2，单位米。
    depth_noise_base_m: float = 0.0015
    depth_noise_quadratic: float = 0.0015
    depth_quantization_m: float = 0.001
    random_dropout_probability: float = 0.005
    edge_dropout_probability: float = 0.10
    edge_threshold_m: float = 0.025


@dataclass
class CameraRuntime:
    spec: CameraSpec
    camera_prim: Any
    render_product: Any
    rgb_annotator: Any
    depth_annotator: Any
    K: np.ndarray
    T_world_from_camera_optical: np.ndarray
    T_camera_optical_from_world: np.ndarray


@dataclass
class CameraFrame:
    runtime: CameraRuntime

    # 实际发布的流；corruption.enabled=True 时为损坏后的数据。
    rgb: np.ndarray
    depth_m: np.ndarray
    points_camera_optical: np.ndarray
    points_world: np.ndarray
    colors: np.ndarray

    # 始终保留干净数据，便于同时发布或实验对比。
    clean_rgb: np.ndarray
    clean_depth_m: np.ndarray
    clean_points_camera_optical: np.ndarray
    clean_points_world: np.ndarray
    clean_colors: np.ndarray


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError("无法归一化近零向量")
    return vector / norm


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """将 3x3 旋转矩阵转换为 [x, y, z, w]。"""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"旋转矩阵尺寸必须为 3x3，当前为 {matrix.shape}")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(
            1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
        ) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(
            1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
        ) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(
            1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
        ) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def make_camera_pose(
    position_world: tuple[float, float, float],
    look_at_world: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """返回 optical 相机到世界的齐次变换，以及 USD 相机 quaternion。"""

    eye = np.asarray(position_world, dtype=np.float64)
    target = np.asarray(look_at_world, dtype=np.float64)
    up = _normalize(np.asarray(world_up, dtype=np.float64))

    z_forward = _normalize(target - eye)
    x_right = _normalize(np.cross(z_forward, up))
    y_down = _normalize(np.cross(z_forward, x_right))

    R_world_from_optical = np.column_stack(
        (x_right, y_down, z_forward)
    )

    # USD Camera 本地坐标：+x 右、+y 上、-z 前。
    optical_from_usd = np.diag([1.0, -1.0, -1.0])
    R_world_from_usd = R_world_from_optical @ optical_from_usd

    T_world_from_optical = np.eye(4, dtype=np.float64)
    T_world_from_optical[:3, :3] = R_world_from_optical
    T_world_from_optical[:3, 3] = eye

    quaternion_usd_xyzw = rotation_matrix_to_quaternion_xyzw(
        R_world_from_usd
    )
    return T_world_from_optical, quaternion_usd_xyzw


def camera_intrinsics(
    width: int,
    height: int,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
) -> tuple[np.ndarray, float]:
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
    x, y, z, w = [float(value) for value in quaternion_usd_xyzw]

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(
        Gf.Vec3d(*[float(value) for value in translation_world])
    )

    # AddOrientOp 默认创建 GfQuatf 属性，因此必须传 Gf.Quatf。
    xformable.AddOrientOp().Set(
        Gf.Quatf(w, Gf.Vec3f(x, y, z))
    )


def create_cameras(
    stage,
    rig: CameraRigConfig,
) -> list[CameraRuntime]:
    if rig.width <= 0 or rig.height <= 0:
        raise ValueError("相机分辨率必须为正数")
    if rig.point_stride <= 0:
        raise ValueError("point_stride 必须大于 0")

    cameras: list[CameraRuntime] = []

    for spec in rig.camera_specs:
        T_world_from_optical, quaternion_usd_xyzw = make_camera_pose(
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
        camera.CreateVerticalApertureAttr(float(vertical_aperture_mm))
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

        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        rgb_annotator.attach(render_product)
        depth_annotator.attach(render_product)

        cameras.append(
            CameraRuntime(
                spec=spec,
                camera_prim=camera,
                render_product=render_product,
                rgb_annotator=rgb_annotator,
                depth_annotator=depth_annotator,
                K=K,
                T_world_from_camera_optical=T_world_from_optical,
                T_camera_optical_from_world=np.linalg.inv(
                    T_world_from_optical
                ),
            )
        )

    return cameras


def _as_array(data: Any, label: str) -> np.ndarray:
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    array = np.asarray(data)
    if array.size == 0:
        raise RuntimeError(f"{label} annotator 返回空数据")
    return array


def corrupt_rgb(
    clean_rgb: np.ndarray,
    config: CorruptionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    image = clean_rgb.astype(np.float32)
    exposure = rng.uniform(
        1.0 - config.exposure_fraction,
        1.0 + config.exposure_fraction,
    )
    image *= exposure
    image += rng.normal(
        0.0,
        config.rgb_noise_std_255,
        size=image.shape,
    ).astype(np.float32)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def corrupt_depth(
    clean_depth_m: np.ndarray,
    config: CorruptionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    depth = clean_depth_m.astype(np.float32, copy=True)
    valid = np.isfinite(depth) & (depth > 0.0)

    sigma = (
        config.depth_noise_base_m
        + config.depth_noise_quadratic * np.square(depth)
    )
    noise = rng.normal(0.0, 1.0, size=depth.shape).astype(np.float32)
    depth[valid] += sigma[valid] * noise[valid]

    if config.depth_quantization_m > 0.0:
        step = float(config.depth_quantization_m)
        depth[valid] = np.round(depth[valid] / step) * step

    dropout = (
        rng.random(depth.shape) < config.random_dropout_probability
    )

    # 深度不连续处的边缘 dropout，用于近似飞点/孔洞。
    grad_x = np.zeros_like(depth)
    grad_y = np.zeros_like(depth)
    grad_x[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
    grad_y[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
    edge = np.maximum(grad_x, grad_y) > config.edge_threshold_m
    edge_dropout = edge & (
        rng.random(depth.shape) < config.edge_dropout_probability
    )

    invalid = dropout | edge_dropout | (~valid) | (depth <= 0.0)
    depth[invalid] = np.nan
    return depth


def depth_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    T_world_from_camera_optical: np.ndarray,
    stride: int,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 optical-frame 点、world-frame 点和对应 RGB。"""

    if depth_m.ndim != 2:
        raise ValueError(f"深度图应为 HxW，当前为 {depth_m.shape}")
    if rgb.ndim != 3 or rgb.shape[:2] != depth_m.shape:
        raise ValueError(
            f"RGB/深度尺寸不匹配：rgb={rgb.shape}, depth={depth_m.shape}"
        )

    depth = depth_m[::stride, ::stride]
    colors_grid = rgb[::stride, ::stride, :3]

    height, width = depth_m.shape
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    z = depth.astype(np.float32, copy=False)

    valid = (
        np.isfinite(z)
        & (z > 0.0)
        & (z < float(max_depth_m))
    )

    z_valid = z[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy

    points_optical = np.stack((x, y, z_valid), axis=1).astype(
        np.float32
    )

    rotation = T_world_from_camera_optical[:3, :3]
    translation = T_world_from_camera_optical[:3, 3]
    points_world = (
        points_optical @ rotation.T + translation[None, :]
    ).astype(np.float32)

    colors = colors_grid[valid].astype(np.uint8, copy=False)
    return points_optical, points_world, colors


def capture_all_cameras(
    cameras: Iterable[CameraRuntime],
    rig: CameraRigConfig,
    corruption: CorruptionConfig,
    rng: np.random.Generator,
) -> dict[str, CameraFrame]:
    frames: dict[str, CameraFrame] = {}

    for runtime in cameras:
        rgb_raw = _as_array(
            runtime.rgb_annotator.get_data(),
            f"{runtime.spec.name}/rgb",
        )
        depth_raw = _as_array(
            runtime.depth_annotator.get_data(),
            f"{runtime.spec.name}/depth",
        )

        if rgb_raw.ndim != 3 or rgb_raw.shape[2] < 3:
            raise RuntimeError(
                f"{runtime.spec.name} RGB 尺寸异常：{rgb_raw.shape}"
            )
        clean_rgb = rgb_raw[..., :3].astype(np.uint8, copy=False)

        clean_depth = depth_raw.astype(np.float32, copy=False)
        if clean_depth.ndim != 2:
            clean_depth = np.squeeze(clean_depth)
        if clean_depth.ndim != 2:
            raise RuntimeError(
                f"{runtime.spec.name} 深度尺寸异常：{depth_raw.shape}"
            )

        clean_points_optical, clean_points_world, clean_colors = (
            depth_to_pointcloud(
                clean_depth,
                clean_rgb,
                runtime.K,
                runtime.T_world_from_camera_optical,
                rig.point_stride,
                rig.max_depth_m,
            )
        )

        if corruption.enabled:
            rgb = corrupt_rgb(clean_rgb, corruption, rng)
            depth = corrupt_depth(clean_depth, corruption, rng)
        else:
            rgb = clean_rgb
            depth = clean_depth

        points_optical, points_world, colors = depth_to_pointcloud(
            depth,
            rgb,
            runtime.K,
            runtime.T_world_from_camera_optical,
            rig.point_stride,
            rig.max_depth_m,
        )

        frames[runtime.spec.name] = CameraFrame(
            runtime=runtime,
            rgb=rgb,
            depth_m=depth,
            points_camera_optical=points_optical,
            points_world=points_world,
            colors=colors,
            clean_rgb=clean_rgb,
            clean_depth_m=clean_depth,
            clean_points_camera_optical=clean_points_optical,
            clean_points_world=clean_points_world,
            clean_colors=clean_colors,
        )

    return frames


def fuse_world_pointcloud(
    frames: dict[str, CameraFrame],
    clean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    if clean:
        points = [
            frame.clean_points_world for frame in frames.values()
        ]
        colors = [frame.clean_colors for frame in frames.values()]
    else:
        points = [frame.points_world for frame in frames.values()]
        colors = [frame.colors for frame in frames.values()]

    return (
        np.concatenate(points, axis=0),
        np.concatenate(colors, axis=0),
    )


def camera_calibration_dict(
    cameras: Iterable[CameraRuntime],
    rig: CameraRigConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "world_frame_id": rig.world_frame_id,
        "world_convention": "Isaac Sim: +z up, meters",
        "camera_convention": "ROS/OpenCV optical: +x right, +y down, +z forward",
        "depth_definition": "distance_to_image_plane in meters",
        "image_width": rig.width,
        "image_height": rig.height,
        "cameras": {},
    }

    for runtime in cameras:
        quaternion = rotation_matrix_to_quaternion_xyzw(
            runtime.T_world_from_camera_optical[:3, :3]
        )
        result["cameras"][runtime.spec.name] = {
            "prim_path": runtime.spec.prim_path,
            "optical_frame_id": runtime.spec.optical_frame_id,
            "K": runtime.K.tolist(),
            "position_world_m": list(runtime.spec.position_world),
            "orientation_world_from_optical_xyzw": quaternion.tolist(),
            "T_world_from_camera_optical":
                runtime.T_world_from_camera_optical.tolist(),
            "T_camera_optical_from_world":
                runtime.T_camera_optical_from_world.tolist(),
            "focal_length_mm": runtime.spec.focal_length_mm,
            "horizontal_aperture_mm":
                runtime.spec.horizontal_aperture_mm,
            "near_m": runtime.spec.near_m,
            "far_m": runtime.spec.far_m,
        }

    return result


def save_calibration(
    output_dir: Path,
    cameras: Iterable[CameraRuntime],
    rig: CameraRigConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibration.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            camera_calibration_dict(cameras, rig),
            file,
            indent=2,
        )


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"点云尺寸异常：{points.shape}")
    if colors.shape != points.shape:
        raise ValueError(
            f"点和颜色尺寸不一致：{points.shape} vs {colors.shape}"
        )

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = (
        points[:, 0],
        points[:, 1],
        points[:, 2],
    )
    vertices["red"], vertices["green"], vertices["blue"] = (
        colors[:, 0],
        colors[:, 1],
        colors[:, 2],
    )

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as file:
        file.write(header)
        vertices.tofile(file)


def save_frame_bundle(
    output_dir: Path,
    frame_index: int,
    frames: dict[str, CameraFrame],
) -> Path:
    frame_dir = output_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in frames.items():
        Image.fromarray(frame.rgb).save(frame_dir / f"{name}_rgb.png")
        np.save(frame_dir / f"{name}_depth.npy", frame.depth_m)
        np.savez_compressed(
            frame_dir / f"{name}_rgbd.npz",
            rgb=frame.rgb,
            depth_m=frame.depth_m,
            K=frame.runtime.K,
            T_world_from_camera_optical=
                frame.runtime.T_world_from_camera_optical,
            points_camera_optical=frame.points_camera_optical,
            points_world=frame.points_world,
            colors=frame.colors,
        )
        write_binary_ply(
            frame_dir / f"{name}_world.ply",
            frame.points_world,
            frame.colors,
        )

    fused_points, fused_colors = fuse_world_pointcloud(frames)
    write_binary_ply(
        frame_dir / "fused_world.ply",
        fused_points,
        fused_colors,
    )
    np.savez_compressed(
        frame_dir / "fused_world.npz",
        points=fused_points,
        colors=fused_colors,
    )
    return frame_dir
