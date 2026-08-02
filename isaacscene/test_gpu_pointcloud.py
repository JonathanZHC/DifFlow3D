#!/usr/bin/env python3
"""Sanity-check the full-resolution per-camera CUDA point-cloud builder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from pointcloud_full_gpu import GpuCameraPointCloudBuilder


def make_frame():
    height, width = 48, 64
    depth_value = 1.25

    depth = np.full(
        (height, width),
        depth_value,
        dtype=np.float32,
    )
    depth[0, 0] = 0.0

    rgb = np.zeros(
        (height, width, 4),
        dtype=np.uint8,
    )
    rgb[..., 0] = 12
    rgb[..., 1] = 34
    rgb[..., 2] = 56
    rgb[..., 3] = 255

    fx = 60.0
    fy = 60.0
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    ray_x = (u - cx) / fx
    ray_y = (v - cy) / fy

    runtime = SimpleNamespace(
        spec=SimpleNamespace(name="camera_0"),
        ray_x=ray_x,
        ray_y=ray_y,
    )

    return SimpleNamespace(
        runtime=runtime,
        depth_gpu=wp.array(
            depth,
            dtype=wp.float32,
            device="cuda:0",
        ),
        rgb_gpu=wp.array(
            rgb,
            dtype=wp.uint8,
            device="cuda:0",
        ),
    )


def main() -> None:
    wp.init()

    builder = GpuCameraPointCloudBuilder(
        max_depth_m=10.0,
        device="cuda:0",
    )
    cloud = builder.build(make_frame())

    expected_points = 48 * 64 - 1
    assert cloud.shape == (expected_points, 4)
    assert cloud.dtype == np.float32
    assert np.isfinite(cloud[:, :3]).all()
    assert np.allclose(cloud[:, 2], 1.25)

    packed = np.ascontiguousarray(
        cloud[:, 3],
    ).view(np.uint32)
    expected_rgb = (12 << 16) | (34 << 8) | 56
    assert np.all(packed == expected_rgb)

    print("Full GPU point-cloud sanity check passed.")
    print("Output points:", cloud.shape[0])
    print("Output bytes:", cloud.nbytes)


if __name__ == "__main__":
    main()
