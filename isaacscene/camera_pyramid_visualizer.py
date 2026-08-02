#!/usr/bin/env python3
"""Draw static camera pyramids in the Isaac Sim viewport.

The pyramids use Isaac Sim Debug Draw, so they are viewport overlays rather
than USD scene geometry and are not captured by the RGB-D cameras.
"""

from __future__ import annotations

import numpy as np

from camera_settings import CameraRuntime


PYRAMID_DEPTH_M = 0.45


def _world_point(
    point_optical: np.ndarray,
    transform: np.ndarray,
) -> tuple[float, float, float]:
    point_world = (
        transform[:3, :3] @ point_optical
        + transform[:3, 3]
    )
    return tuple(float(value) for value in point_world)


def _frustum_corners(
    camera: CameraRuntime,
    width: int,
    height: int,
) -> list[np.ndarray]:
    K = camera.K
    depth = PYRAMID_DEPTH_M

    def project(u: float, v: float) -> np.ndarray:
        return np.asarray(
            [
                (u - float(K[0, 2])) * depth / float(K[0, 0]),
                (v - float(K[1, 2])) * depth / float(K[1, 1]),
                depth,
            ],
            dtype=np.float64,
        )

    return [
        project(0.0, 0.0),
        project(float(width - 1), 0.0),
        project(float(width - 1), float(height - 1)),
        project(0.0, float(height - 1)),
    ]


class CameraPyramidVisualizer:
    """Own the persistent Debug Draw camera pyramids."""

    def __init__(
        self,
        cameras: list[CameraRuntime],
        width: int,
        height: int,
    ) -> None:
        from isaacsim.util.debug_draw import _debug_draw

        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._draw.clear_lines()
        self._draw.clear_points()

        palette = (
            (0.10, 0.85, 1.00, 1.0),
            (1.00, 0.55, 0.10, 1.0),
        )

        line_starts: list[tuple[float, float, float]] = []
        line_ends: list[tuple[float, float, float]] = []
        line_colors: list[tuple[float, float, float, float]] = []
        line_widths: list[float] = []
        camera_positions: list[tuple[float, float, float]] = []
        camera_colors: list[tuple[float, float, float, float]] = []

        for index, camera in enumerate(cameras):
            transform = camera.T_world_from_camera_optical
            origin = tuple(
                float(value) for value in transform[:3, 3]
            )
            corners = [
                _world_point(point, transform)
                for point in _frustum_corners(
                    camera,
                    width,
                    height,
                )
            ]
            color = palette[index % len(palette)]

            segments = [
                (origin, corners[0]),
                (origin, corners[1]),
                (origin, corners[2]),
                (origin, corners[3]),
                (corners[0], corners[1]),
                (corners[1], corners[2]),
                (corners[2], corners[3]),
                (corners[3], corners[0]),
            ]

            for start, end in segments:
                line_starts.append(start)
                line_ends.append(end)
                line_colors.append(color)
                line_widths.append(3.0)

            camera_positions.append(origin)
            camera_colors.append(color)

        self._draw.draw_lines(
            line_starts,
            line_ends,
            line_colors,
            line_widths,
        )
        self._draw.draw_points(
            camera_positions,
            camera_colors,
            [12.0] * len(camera_positions),
        )

    def clear(self) -> None:
        self._draw.clear_lines()
        self._draw.clear_points()
