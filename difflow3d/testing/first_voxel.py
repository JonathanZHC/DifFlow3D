"""First-stage metric voxel preprocessing for benchmark sensor clouds."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch

@dataclass(frozen=True)
class FirstVoxelFrame:
    raw_points_cpu: np.ndarray
    first_downsample_points: torch.Tensor
    first_raw_indices: torch.Tensor
    timestamp_s: float
    raw_count: int
    first_count: int
    host_stage_ms: float
    h2d_ms: float
    first_downsample_ms: float
    preprocess_gpu_ms: float
    preprocess_wall_ms: float

class FirstVoxelPreprocessor:
    """CPU/pinned H2D + mandatory first GPU voxel only.

    Voxel-2, fixed-count selection and spatial scaling intentionally live inside
    difflow3d.runtime.DifFlow3DStreamingCudaGraphRunner.
    """

    def __init__(
        self,
        *,
        raw_point_count: int,
        device: torch.device,
        first_voxel_size_m: float,
    ) -> None:
        if raw_point_count < 1:
            raise ValueError("raw_point_count must be positive.")
        if first_voxel_size_m <= 0.0:
            raise ValueError("first_voxel_size_m must be positive.")
        self.raw_point_count = int(raw_point_count)
        self.device = device
        self.first_voxel_size_m = float(first_voxel_size_m)

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

    @staticmethod
    def _voxel_downsample(
        points: torch.Tensor,
        voxel_size_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = torch.floor(points / voxel_size_m).to(torch.int64)
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

    def process(self, points: np.ndarray, timestamp_s: float) -> FirstVoxelFrame:
        array = np.asarray(points, dtype=np.float32)
        if array.shape != (self.raw_point_count, 3):
            raise ValueError(
                f"Expected raw cloud {(self.raw_point_count, 3)}, got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Raw point cloud contains NaN or Inf values.")

        wall_start = time.perf_counter()
        host_start = time.perf_counter()
        np.copyto(self._host_np, array)
        host_stage_ms = 1000.0 * (time.perf_counter() - host_start)

        event_start = torch.cuda.Event(enable_timing=True)
        event_h2d = torch.cuda.Event(enable_timing=True)
        event_first = torch.cuda.Event(enable_timing=True)
        event_start.record()
        self._device_raw.copy_(self._host, non_blocking=True)
        event_h2d.record()
        first_points, first_local_indices = self._voxel_downsample(
            self._device_raw,
            self.first_voxel_size_m,
        )
        first_raw_indices = self._raw_indices.index_select(0, first_local_indices)
        event_first.record()
        event_first.synchronize()

        h2d_ms = float(event_start.elapsed_time(event_h2d))
        first_downsample_ms = float(event_h2d.elapsed_time(event_first))
        preprocess_gpu_ms = float(event_start.elapsed_time(event_first))
        preprocess_wall_ms = 1000.0 * (time.perf_counter() - wall_start)

        return FirstVoxelFrame(
            raw_points_cpu=array,
            first_downsample_points=first_points,
            first_raw_indices=first_raw_indices,
            timestamp_s=float(timestamp_s),
            raw_count=self.raw_point_count,
            first_count=int(first_points.shape[0]),
            host_stage_ms=host_stage_ms,
            h2d_ms=h2d_ms,
            first_downsample_ms=first_downsample_ms,
            preprocess_gpu_ms=preprocess_gpu_ms,
            preprocess_wall_ms=preprocess_wall_ms,
        )
