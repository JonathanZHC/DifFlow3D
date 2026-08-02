#!/usr/bin/env python3
"""Compile and sanity-check the camera corruption Warp kernels."""

from __future__ import annotations

import numpy as np
import warp as wp

from camera_corruption_warp import (
    depth_camera_corruption_wp,
    rgb_camera_corruption_wp,
)


def main() -> None:
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA is unavailable.")

    device = "cuda:0"
    height, width = 48, 64
    rng = np.random.default_rng(7)

    rgb_np = rng.integers(
        0,
        256,
        size=(height, width, 4),
        dtype=np.uint8,
    )
    rgb_np[..., 3] = 255
    rgb_in = wp.array(
        rgb_np,
        dtype=wp.uint8,
        device=device,
    )
    rgb_out = wp.empty(
        rgb_np.shape,
        dtype=wp.uint8,
        device=device,
    )

    depth_np = np.full(
        (height, width),
        1.0,
        dtype=np.float32,
    )
    depth_np[:, width // 2 :] = 1.5
    depth_np[0, 0] = 0.0
    depth_in = wp.array(
        depth_np,
        dtype=wp.float32,
        device=device,
    )
    depth_out = wp.empty(
        depth_np.shape,
        dtype=wp.float32,
        device=device,
    )

    wp.launch(
        rgb_camera_corruption_wp,
        dim=(height, width),
        inputs=[rgb_in, rgb_out, 2.0, 0.10, 7],
        device=device,
    )
    wp.launch(
        depth_camera_corruption_wp,
        dim=(height, width),
        inputs=[
            depth_in,
            depth_out,
            0.0015,
            0.0015,
            0.001,
            0.01,
            0.25,
            0.025,
            11,
        ],
        device=device,
    )
    wp.synchronize()

    rgb_result = rgb_out.numpy()
    depth_result = depth_out.numpy()

    assert rgb_result.shape == rgb_np.shape
    assert np.any(rgb_result[..., :3] != rgb_np[..., :3])
    assert np.isnan(depth_result[0, 0])
    assert np.isfinite(depth_result).any()

    print("Warp camera corruption sanity check passed.")
    print("Device:", device)
    print(
        "RGB changed values:",
        int(np.count_nonzero(rgb_result != rgb_np)),
    )
    print(
        "Valid depth pixels:",
        int(np.count_nonzero(np.isfinite(depth_result))),
    )


if __name__ == "__main__":
    main()
