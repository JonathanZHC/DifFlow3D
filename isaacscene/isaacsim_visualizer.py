#!/usr/bin/env python3
"""Isaac Sim GUI 内部可视化。

功能：
  1. 在主 Stage 中显示当前主融合点云；
  2. 显示两台相机的位置、机身、坐标轴和视锥；
  3. 在 Isaac Sim 内创建两个图像窗口，显示当前主 RGB 流。

主流定义：
  - --corrupt 时为损坏数据；
  - 未使用 --corrupt 时为干净数据。

为了不进一步拖慢 RGB-D 发布，点云和图像窗口按独立低频更新，
并限制在 GUI 中显示的最大点数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import omni.ui as ui
from pxr import Gf, UsdGeom, Vt

from camera_settings import (
    CameraFrame,
    CameraRuntime,
    rotation_matrix_to_quaternion_xyzw,
)


@dataclass
class _ImageWindow:
    window: Any
    provider: Any
    label: Any


def _to_vec3f_array(values: np.ndarray) -> Vt.Vec3fArray:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return Vt.Vec3fArray.FromNumpy(array)


def _set_display_color(
    gprim,
    colors: np.ndarray,
    interpolation=UsdGeom.Tokens.vertex,
) -> Any:
    primvar = gprim.CreateDisplayColorPrimvar(interpolation)
    primvar.Set(_to_vec3f_array(colors))
    return primvar


def _transform_points(
    points_local: np.ndarray,
    transform_world_from_local: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_local, dtype=np.float64)
    rotation = transform_world_from_local[:3, :3]
    translation = transform_world_from_local[:3, 3]
    return points @ rotation.T + translation[None, :]


def _frustum_corners_optical(
    runtime: CameraRuntime,
    width: int,
    height: int,
    depth_m: float,
) -> np.ndarray:
    K = runtime.K
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    pixels = np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )

    x = (pixels[:, 0] - cx) * depth_m / fx
    y = (pixels[:, 1] - cy) * depth_m / fy
    z = np.full(4, depth_m, dtype=np.float64)
    return np.column_stack((x, y, z))


def _define_line_segments(
    stage,
    path: str,
    segments_world: list[tuple[np.ndarray, np.ndarray]],
    color: tuple[float, float, float],
    width_m: float,
) -> UsdGeom.BasisCurves:
    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)

    points: list[np.ndarray] = []
    counts: list[int] = []
    for start, end in segments_world:
        points.extend((start, end))
        counts.append(2)

    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray(counts))
    curves.CreatePointsAttr().Set(
        _to_vec3f_array(np.asarray(points, dtype=np.float32))
    )
    curves.CreateWidthsAttr().Set(
        Vt.FloatArray([float(width_m)])
    )
    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    _set_display_color(
        curves,
        np.asarray([color], dtype=np.float32),
        interpolation=UsdGeom.Tokens.constant,
    )
    return curves


class IsaacSimVisualizer:
    """管理 Isaac Sim Stage 点云、相机几何和图像窗口。"""

    def __init__(
        self,
        stage,
        cameras: list[CameraRuntime],
        image_width: int,
        image_height: int,
        primary_stream_label: str,
        update_hz: float = 5.0,
        max_points: int = 40000,
        point_size_m: float = 0.008,
        frustum_depth_m: float = 0.45,
        show_image_windows: bool = True,
    ) -> None:
        if update_hz <= 0.0:
            raise ValueError("Isaac 可视化 update_hz 必须大于 0")
        if max_points <= 0:
            raise ValueError("Isaac 可视化 max_points 必须大于 0")
        if point_size_m <= 0.0:
            raise ValueError("Isaac 可视化 point_size_m 必须大于 0")
        if frustum_depth_m <= 0.0:
            raise ValueError("Isaac 可视化 frustum_depth_m 必须大于 0")

        self.stage = stage
        self.cameras = cameras
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.primary_stream_label = primary_stream_label.upper()
        self.update_period_s = 1.0 / float(update_hz)
        self.max_points = int(max_points)
        self.point_size_m = float(point_size_m)
        self.frustum_depth_m = float(frustum_depth_m)
        self.show_image_windows = bool(show_image_windows)
        self._next_update_time = 0.0

        UsdGeom.Xform.Define(stage, "/World/Visualizations")
        UsdGeom.Xform.Define(
            stage, "/World/Visualizations/Cameras"
        )

        self.points_prim = UsdGeom.Points.Define(
            stage,
            "/World/Visualizations/PrimaryFusedPointCloud",
        )
        self.points_attr = self.points_prim.CreatePointsAttr()
        self.widths_attr = self.points_prim.CreateWidthsAttr()
        self.widths_attr.Set(
            Vt.FloatArray([self.point_size_m])
        )
        self.points_prim.SetWidthsInterpolation(
            UsdGeom.Tokens.constant
        )
        self.color_primvar = (
            self.points_prim.CreateDisplayColorPrimvar(
                UsdGeom.Tokens.vertex
            )
        )

        self.image_windows: dict[str, _ImageWindow] = {}

        self._create_camera_geometry()
        if self.show_image_windows:
            self._create_image_windows()

        print(
            "[IsaacViz] 已启动："
            f" stream={self.primary_stream_label},"
            f" update_hz={update_hz},"
            f" max_points={self.max_points}",
            flush=True,
        )

    def _create_camera_geometry(self) -> None:
        palette = (
            (0.05, 0.75, 1.00),
            (1.00, 0.40, 0.05),
            (0.65, 0.20, 1.00),
            (0.20, 1.00, 0.35),
        )

        for index, runtime in enumerate(self.cameras):
            name = runtime.spec.name
            root = f"/World/Visualizations/Cameras/{name}"
            UsdGeom.Xform.Define(self.stage, root)

            transform = runtime.T_world_from_camera_optical
            origin = transform[:3, 3].astype(np.float64)

            # 相机机身。姿态使用 optical frame 旋转。
            body = UsdGeom.Cube.Define(
                self.stage,
                f"{root}/Body",
            )
            body.CreateSizeAttr(1.0)
            quaternion = rotation_matrix_to_quaternion_xyzw(
                transform[:3, :3]
            )
            x, y, z, w = [float(v) for v in quaternion]

            xformable = UsdGeom.Xformable(body.GetPrim())
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(
                Gf.Vec3d(*[float(v) for v in origin])
            )
            xformable.AddOrientOp().Set(
                Gf.Quatf(w, Gf.Vec3f(x, y, z))
            )
            xformable.AddScaleOp().Set(
                Gf.Vec3f(0.10, 0.065, 0.055)
            )
            _set_display_color(
                body,
                np.asarray(
                    [palette[index % len(palette)]],
                    dtype=np.float32,
                ),
                interpolation=UsdGeom.Tokens.constant,
            )

            corners_optical = _frustum_corners_optical(
                runtime,
                self.image_width,
                self.image_height,
                self.frustum_depth_m,
            )
            corners_world = _transform_points(
                corners_optical,
                transform,
            )
            tl, tr, br, bl = corners_world

            frustum_segments = [
                (origin, tl),
                (origin, tr),
                (origin, br),
                (origin, bl),
                (tl, tr),
                (tr, br),
                (br, bl),
                (bl, tl),
            ]
            _define_line_segments(
                self.stage,
                f"{root}/Frustum",
                frustum_segments,
                palette[index % len(palette)],
                width_m=0.008,
            )

            # optical frame 轴：x 红，y 绿，z 蓝。
            rotation = transform[:3, :3]
            axis_length = 0.22
            axes = (
                (
                    "AxisX",
                    origin,
                    origin + rotation[:, 0] * axis_length,
                    (1.0, 0.0, 0.0),
                ),
                (
                    "AxisY",
                    origin,
                    origin + rotation[:, 1] * axis_length,
                    (0.0, 1.0, 0.0),
                ),
                (
                    "AxisZ",
                    origin,
                    origin + rotation[:, 2] * axis_length,
                    (0.0, 0.25, 1.0),
                ),
            )
            for axis_name, start, end, color in axes:
                _define_line_segments(
                    self.stage,
                    f"{root}/{axis_name}",
                    [(start, end)],
                    color,
                    width_m=0.012,
                )

    def _create_image_windows(self) -> None:
        window_width = min(self.image_width + 24, 680)
        window_height = min(self.image_height + 70, 560)

        for runtime in self.cameras:
            name = runtime.spec.name
            window = ui.Window(
                f"{name} — {self.primary_stream_label} RGB",
                width=window_width,
                height=window_height,
                visible=True,
            )
            provider = ui.ByteImageProvider()

            with window.frame:
                with ui.VStack(spacing=4):
                    label = ui.Label(
                        f"{name} | {self.primary_stream_label}",
                        height=24,
                        alignment=ui.Alignment.CENTER,
                    )
                    ui.ImageWithProvider(
                        provider,
                        fill_policy=(
                            ui.IwpFillPolicy
                            .IWP_PRESERVE_ASPECT_FIT
                        ),
                    )

            self.image_windows[name] = _ImageWindow(
                window=window,
                provider=provider,
                label=label,
            )

    def _sample_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = int(points.shape[0])
        if count <= self.max_points:
            return points, colors

        # 均匀索引抽样，确定性且不需要每帧随机数生成。
        indices = np.linspace(
            0,
            count - 1,
            self.max_points,
            dtype=np.int64,
        )
        return points[indices], colors[indices]

    def _update_point_cloud(
        self,
        points_world: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        points, sampled_colors = self._sample_points(
            np.asarray(points_world, dtype=np.float32),
            np.asarray(colors, dtype=np.uint8),
        )

        if points.shape[0] == 0:
            self.points_attr.Set(Vt.Vec3fArray())
            self.color_primvar.Set(Vt.Vec3fArray())
            return

        normalized_colors = (
            sampled_colors.astype(np.float32) / 255.0
        )
        self.points_attr.Set(_to_vec3f_array(points))
        self.color_primvar.Set(
            _to_vec3f_array(normalized_colors)
        )

    @staticmethod
    def _rgb_to_rgba(rgb: np.ndarray) -> np.ndarray:
        rgb_u8 = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width = rgb_u8.shape[:2]
        rgba = np.empty(
            (height, width, 4),
            dtype=np.uint8,
        )
        rgba[..., :3] = rgb_u8[..., :3]
        rgba[..., 3] = 255
        return rgba

    def _update_image_windows(
        self,
        frames: dict[str, CameraFrame],
    ) -> None:
        for name, image_window in self.image_windows.items():
            if name not in frames:
                continue

            frame = frames[name]
            rgba = self._rgb_to_rgba(frame.rgb)
            height, width = rgba.shape[:2]

            # set_data_array 避免把 NumPy 数组转成巨大的 Python list。
            image_window.provider.set_data_array(
                rgba.reshape(-1),
                [int(width), int(height)],
            )
            image_window.label.text = (
                f"{name} | {self.primary_stream_label} | "
                f"{width}×{height}"
            )

    def update(
        self,
        frames: dict[str, CameraFrame],
        fused_points_world: np.ndarray,
        fused_colors: np.ndarray,
    ) -> None:
        now = time.perf_counter()
        if now < self._next_update_time:
            return
        self._next_update_time = now + self.update_period_s

        self._update_point_cloud(
            fused_points_world,
            fused_colors,
        )
        if self.show_image_windows:
            self._update_image_windows(frames)

    def shutdown(self) -> None:
        for image_window in self.image_windows.values():
            try:
                image_window.provider.destroy()
            except Exception:
                pass
            try:
                image_window.window.destroy()
            except Exception:
                pass
        self.image_windows.clear()
