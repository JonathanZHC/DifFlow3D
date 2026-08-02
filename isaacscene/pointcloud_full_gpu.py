#!/usr/bin/env python3
"""Build one full-resolution PointCloud2 payload per camera on CUDA.

The input RGB and depth arrays are the active Isaac Sim CUDA streams. They may
be clean or GPU-corrupted, depending on the command-line configuration.

The GPU path performs:

1. Valid-depth filtering.
2. Depth back-projection into the camera optical frame.
3. RGB packing for PointCloud2.

There is no voxelization, sampling, point limit, world transformation, camera
fusion, sorting, or duplicate removal. Only the final valid XYZRGB array is
copied to CPU memory for rclpy PointCloud2 serialization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp


class GpuCameraPointCloudBuilder:
    """Build full valid XYZRGB clouds in each camera optical frame."""

    def __init__(
        self,
        max_depth_m: float,
        device: str = "cuda:0",
    ) -> None:
        if max_depth_m <= 0.0:
            raise ValueError("max_depth_m must be positive.")
        if not wp.is_cuda_available():
            raise RuntimeError("CUDA is unavailable to NVIDIA Warp.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to PyTorch.")

        self.max_depth_m = float(max_depth_m)
        self.device = device
        self.torch_device = torch.device(
            wp.device_to_torch(device)
        )

        self._ray_cache: dict[
            str,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def _camera_rays(
        self,
        runtime: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened optical-frame ray multipliers on CUDA."""

        name = runtime.spec.name
        cached = self._ray_cache.get(name)
        if cached is not None:
            return cached

        ray_x = torch.as_tensor(
            np.ascontiguousarray(
                runtime.ray_x,
                dtype=np.float32,
            ),
            device=self.torch_device,
            dtype=torch.float32,
        ).reshape(-1)

        ray_y = torch.as_tensor(
            np.ascontiguousarray(
                runtime.ray_y,
                dtype=np.float32,
            ),
            device=self.torch_device,
            dtype=torch.float32,
        ).reshape(-1)

        cached = (ray_x, ray_y)
        self._ray_cache[name] = cached
        return cached

    def build(
        self,
        frame: Any,
    ) -> np.ndarray:
        """Return an Nx4 float32 XYZRGB array in the optical frame.

        The fourth float stores the packed RGB bit pattern expected by the
        standard PointCloud2 "rgb" FLOAT32 field.
        """

        with torch.inference_mode():
            depth = wp.to_torch(frame.depth_gpu).squeeze()
            rgb = wp.to_torch(frame.rgb_gpu)

            if depth.ndim != 2:
                raise RuntimeError(
                    f"{frame.runtime.spec.name} depth shape is "
                    f"{tuple(depth.shape)}."
                )
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise RuntimeError(
                    f"{frame.runtime.spec.name} RGB shape is "
                    f"{tuple(rgb.shape)}."
                )
            if tuple(rgb.shape[:2]) != tuple(depth.shape):
                raise RuntimeError(
                    f"{frame.runtime.spec.name} RGB/depth mismatch: "
                    f"{tuple(rgb.shape)} vs {tuple(depth.shape)}."
                )

            ray_x, ray_y = self._camera_rays(frame.runtime)
            z_all = depth.reshape(-1)

            if ray_x.numel() != z_all.numel():
                raise RuntimeError(
                    f"{frame.runtime.spec.name} ray/depth mismatch: "
                    f"{ray_x.numel()} vs {z_all.numel()}."
                )

            valid = (
                torch.isfinite(z_all)
                & (z_all > 0.0)
                & (z_all < self.max_depth_m)
            )
            valid_indices = torch.nonzero(
                valid,
                as_tuple=False,
            ).squeeze(1)

            point_count = int(valid_indices.numel())
            if point_count == 0:
                return np.empty((0, 4), dtype=np.float32)

            z = z_all.index_select(0, valid_indices)
            x = ray_x.index_select(0, valid_indices) * z
            y = ray_y.index_select(0, valid_indices) * z

            colors = (
                rgb[..., :3]
                .reshape(-1, 3)
                .index_select(0, valid_indices)
                .to(dtype=torch.int32)
            )

            packed_rgb = (
                (colors[:, 0] << 16)
                | (colors[:, 1] << 8)
                | colors[:, 2]
            ).contiguous()

            cloud_gpu = torch.empty(
                (point_count, 4),
                device=self.torch_device,
                dtype=torch.float32,
            )
            cloud_gpu[:, 0] = x
            cloud_gpu[:, 1] = y
            cloud_gpu[:, 2] = z

            # Preserve the packed 0x00RRGGBB bit pattern in a FLOAT32 field.
            cloud_gpu[:, 3] = packed_rgb.view(torch.float32)

            # This is the only point-cloud device-to-host transfer.
            cloud_cpu = cloud_gpu.cpu().numpy()

        return np.ascontiguousarray(
            cloud_cpu,
            dtype=np.float32,
        )
