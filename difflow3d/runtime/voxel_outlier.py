"""CUDA sparse-voxel spatial/temporal outlier filtering."""

from __future__ import annotations

import math

import torch

from difflow3d.ops.pointnet2 import pointnet2_utils


_STATISTIC_NAMES = (
    "component_count",
    "largest_component_voxels",
    "tiny_removed_component_count",
    "temporal_candidate_component_count",
    "temporal_supported_component_count",
    "temporal_rejected_component_count",
    "removed_component_count",
    "removed_voxel_count",
    "temporal_supported_voxel_count",
    "temporal_candidate_voxel_count",
    "temporal_bypassed_component_count",
)


def resolve_voxel_outlier_statistics(info: dict[str, object]) -> dict[str, object]:
    """Resolve deferred CUDA statistics after the caller already synchronized."""
    statistics = info.pop("_outlier_filter_statistics", None)
    if statistics is None:
        return info
    if not isinstance(statistics, torch.Tensor):
        raise TypeError("Deferred outlier statistics must be a CUDA tensor.")
    values = statistics.detach().cpu().tolist()
    resolved = {
        name: int(value)
        for name, value in zip(_STATISTIC_NAMES, values, strict=True)
    }
    history_available = bool(info.get("outlier_filter_history_available", False))
    candidate_voxels = resolved["temporal_candidate_voxel_count"]
    resolved["temporal_supported_fraction"] = (
        float(resolved["temporal_supported_voxel_count"] / candidate_voxels)
        if history_available and candidate_voxels > 0
        else None
    )
    info["outlier_filter_statistics"] = resolved
    return info


class CudaVoxelComponentOutlierFilter:
    """Filter 26-connected components using size and one-frame support.

    There is no track input at this preprocessing stage, so all voxel-2
    representatives form one virtual object. The unfiltered current voxel set
    becomes the next call's history.
    """

    def __init__(
        self,
        *,
        tiny_component_max_voxels: int,
        max_small_component_fraction: float,
        support_radius_voxels: int,
        min_supported_fraction: float,
    ) -> None:
        if tiny_component_max_voxels < 0:
            raise ValueError("tiny_component_max_voxels must be non-negative.")
        if not 0.0 <= max_small_component_fraction <= 1.0:
            raise ValueError(
                "max_small_component_fraction must be between zero and one."
            )
        if support_radius_voxels < 0:
            raise ValueError("support_radius_voxels must be non-negative.")
        if not 0.0 <= min_supported_fraction <= 1.0:
            raise ValueError("min_supported_fraction must be between zero and one.")

        self.tiny_component_max_voxels = int(tiny_component_max_voxels)
        self.max_small_component_fraction = float(max_small_component_fraction)
        self.support_radius_voxels = int(support_radius_voxels)
        self.min_supported_fraction = float(min_supported_fraction)

        self._capacity = 0
        self._device: torch.device | None = None
        self._parents: torch.Tensor | None = None
        self._component_sizes: torch.Tensor | None = None
        self._supported_counts: torch.Tensor | None = None
        self._largest_component_size: torch.Tensor | None = None
        self._keep_mask: torch.Tensor | None = None
        self._statistics: torch.Tensor | None = None
        self._history_capacity = 0
        self._history_count = 0
        self._history_coords: torch.Tensor | None = None

    def reset_history(self) -> None:
        """Forget temporal support while retaining allocated GPU workspaces."""
        self._history_count = 0

    def _ensure_capacity(self, count: int, device: torch.device) -> None:
        if device != self._device:
            self._capacity = 0
            self._history_capacity = 0
            self._history_count = 0
            self._history_coords = None
        if count > self._capacity or device != self._device:
            capacity = max(count, max(64, self._capacity * 2))
            self._parents = torch.empty(capacity, device=device, dtype=torch.int32)
            self._component_sizes = torch.empty_like(self._parents)
            self._supported_counts = torch.empty_like(self._parents)
            self._largest_component_size = torch.empty(
                1, device=device, dtype=torch.int32
            )
            self._keep_mask = torch.empty(capacity, device=device, dtype=torch.bool)
            self._statistics = torch.empty(11, device=device, dtype=torch.int32)
            self._capacity = int(capacity)
            self._device = device

        if count > self._history_capacity:
            capacity = max(count, max(64, self._history_capacity * 2))
            history = torch.empty((capacity, 3), device=device, dtype=torch.int32)
            if self._history_coords is not None and self._history_count > 0:
                history[: self._history_count].copy_(
                    self._history_coords[: self._history_count]
                )
            self._history_coords = history
            self._history_capacity = int(capacity)

    @torch.no_grad()
    def filter(
        self,
        points: torch.Tensor,
        shifted_coords: torch.Tensor,
        absolute_coords: torch.Tensor,
        sorted_keys: torch.Tensor,
        extents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        count = int(points.shape[0])
        if count < 1:
            raise ValueError("voxel outlier filtering requires at least one point")
        if (
            shifted_coords.shape != (count, 3)
            or absolute_coords.shape != (count, 3)
            or sorted_keys.shape != (count,)
        ):
            raise ValueError("voxel metadata does not match the candidate cloud")

        self._ensure_capacity(count, points.device)
        assert self._parents is not None
        assert self._component_sizes is not None
        assert self._supported_counts is not None
        assert self._largest_component_size is not None
        assert self._keep_mask is not None
        assert self._statistics is not None
        assert self._history_coords is not None

        max_small_component_voxels = max(
            self.tiny_component_max_voxels,
            int(math.ceil(self.max_small_component_fraction * count)),
        )
        history_count = self._history_count
        previous_coords = self._history_coords[:history_count]
        keep_mask = pointnet2_utils.voxel_component_keep_mask(
            sorted_keys,
            shifted_coords,
            absolute_coords,
            extents,
            previous_coords,
            tiny_component_max_voxels=self.tiny_component_max_voxels,
            max_small_component_voxels=max_small_component_voxels,
            support_radius_voxels=self.support_radius_voxels,
            min_supported_fraction=self.min_supported_fraction,
            parents=self._parents[:count],
            component_sizes=self._component_sizes[:count],
            supported_counts=self._supported_counts[:count],
            largest_component_size=self._largest_component_size,
            keep_mask=self._keep_mask[:count],
            statistics=self._statistics,
        )

        # Preserve unfiltered observations: a real new small region can survive
        # its second frame, while a one-frame transient receives no future vote.
        self._history_coords[:count].copy_(absolute_coords)
        self._history_count = count

        retained_indices = torch.nonzero(
            keep_mask, as_tuple=False
        ).flatten().long().contiguous()
        filtered = points.index_select(0, retained_indices).contiguous()
        retained_count = int(filtered.shape[0])
        return filtered, retained_indices, {
            "outlier_filter_input_count": count,
            "outlier_filter_retained_count": retained_count,
            "outlier_filter_removed_count": count - retained_count,
            "outlier_filter_max_small_component_voxels": (
                max_small_component_voxels
            ),
            "outlier_filter_history_available": history_count > 0,
            "_outlier_filter_statistics": self._statistics,
        }
