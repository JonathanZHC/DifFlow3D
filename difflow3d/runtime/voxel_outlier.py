"""Track-aware CUDA sparse-voxel component outlier filtering."""

from __future__ import annotations

import torch

from difflow3d.ops.pointnet2 import pointnet2_utils


_STATISTIC_NAMES = (
    "component_count",
    "largest_component_voxels",
    "retained_component_count",
    "removed_component_count",
    "removed_voxel_count",
    "instance_count",
    "instances_with_removed_components",
    "instances_all_components_retained",
)


def resolve_voxel_outlier_statistics(info: dict[str, object]) -> dict[str, object]:
    """Resolve deferred CUDA statistics after the caller already synchronized."""
    statistics = info.pop("_outlier_filter_statistics", None)
    if statistics is None:
        return info
    if not isinstance(statistics, torch.Tensor):
        raise TypeError("Deferred outlier statistics must be a CUDA tensor.")
    values = statistics.detach().cpu().tolist()
    info["outlier_filter_statistics"] = {
        name: int(value)
        for name, value in zip(_STATISTIC_NAMES, values, strict=True)
    }
    return info


class CudaVoxelComponentOutlierFilter:
    """Keep track-local blocks relative to the largest block in each track.

    For each track with largest component size ``S_max``, a component of size
    ``S`` is retained exactly when
    ``S > S_max * min_component_size_ratio``. All other components are
    rejected. When IDs are unavailable, every voxel belongs to one virtual
    track. The filter has no temporal state and performs only O(N) tensor work
    after CUDA connected-component labeling.
    """

    def __init__(self, *, min_component_size_ratio: float) -> None:
        if not 0.0 <= min_component_size_ratio < 1.0:
            raise ValueError("min_component_size_ratio must be in [0, 1).")
        self.min_component_size_ratio = float(min_component_size_ratio)

        self._capacity = 0
        self._device: torch.device | None = None
        self._parents: torch.Tensor | None = None
        self._component_sizes: torch.Tensor | None = None
        self._op_supported_counts: torch.Tensor | None = None
        self._op_largest_size: torch.Tensor | None = None
        self._op_keep_mask: torch.Tensor | None = None
        self._op_statistics: torch.Tensor | None = None
        self._empty_history: torch.Tensor | None = None
        self._track_max_sizes: torch.Tensor | None = None
        self._removed_components_per_track: torch.Tensor | None = None

    def _ensure_capacity(self, count: int, device: torch.device) -> None:
        if count <= self._capacity and device == self._device:
            return
        capacity = max(count, 64, self._capacity * 2)
        self._parents = torch.empty(capacity, device=device, dtype=torch.int32)
        self._component_sizes = torch.empty_like(self._parents)
        self._op_supported_counts = torch.empty_like(self._parents)
        self._op_largest_size = torch.empty(
            1, device=device, dtype=torch.int32
        )
        self._op_keep_mask = torch.empty(
            capacity, device=device, dtype=torch.bool
        )
        self._op_statistics = torch.empty(
            max(11, len(_STATISTIC_NAMES)), device=device, dtype=torch.int32
        )
        self._empty_history = torch.empty((0, 3), device=device, dtype=torch.int32)
        self._track_max_sizes = torch.empty(
            capacity, device=device, dtype=torch.int64
        )
        self._removed_components_per_track = torch.empty(
            capacity, device=device, dtype=torch.int32
        )
        self._capacity = capacity
        self._device = device

    @torch.no_grad()
    def filter(
        self,
        points: torch.Tensor,
        component_coords: torch.Tensor,
        absolute_coords: torch.Tensor,
        sorted_component_keys: torch.Tensor,
        component_extents: torch.Tensor,
        track_ranks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        count = int(points.shape[0])
        if count < 1:
            raise ValueError("voxel outlier filtering requires at least one point")
        if (
            component_coords.shape != (count, 3)
            or absolute_coords.shape != (count, 3)
            or sorted_component_keys.shape != (count,)
            or track_ranks.shape != (count,)
        ):
            raise ValueError("voxel metadata does not match the candidate cloud")

        self._ensure_capacity(count, points.device)
        assert self._parents is not None
        assert self._component_sizes is not None
        assert self._op_supported_counts is not None
        assert self._op_largest_size is not None
        assert self._op_keep_mask is not None
        assert self._op_statistics is not None
        assert self._empty_history is not None
        assert self._track_max_sizes is not None
        assert self._removed_components_per_track is not None

        parents = self._parents[:count]
        component_sizes = self._component_sizes[:count]

        # Reuse the existing CUDA operator for track-separated 26-neighbor
        # labels and root component sizes. Its legacy decision is configured as
        # a no-op and replaced by the relative-size rule below.
        pointnet2_utils.voxel_component_keep_mask(
            sorted_component_keys,
            component_coords,
            absolute_coords,
            component_extents,
            self._empty_history,
            tiny_component_max_voxels=0,
            max_small_component_voxels=0,
            support_radius_voxels=0,
            min_supported_fraction=0.0,
            parents=parents,
            component_sizes=component_sizes,
            supported_counts=self._op_supported_counts[:count],
            largest_component_size=self._op_largest_size,
            keep_mask=self._op_keep_mask[:count],
            statistics=self._op_statistics,
        )

        indices = torch.arange(count, device=points.device, dtype=torch.int64)
        parent_indices = parents.to(torch.int64)
        roots = parent_indices == indices
        root_sizes = torch.where(
            roots,
            component_sizes.to(torch.int64),
            torch.zeros(count, device=points.device, dtype=torch.int64),
        )

        track_ranks = track_ranks.to(torch.int64)
        track_count = track_ranks.max() + 1
        track_max_sizes = self._track_max_sizes[:count]
        track_max_sizes.zero_()
        track_max_sizes.scatter_reduce_(
            0,
            track_ranks,
            root_sizes,
            reduce="amax",
            include_self=True,
        )
        root_keep = roots & (
            root_sizes.to(torch.float32)
            > track_max_sizes.index_select(0, track_ranks).to(torch.float32)
            * self.min_component_size_ratio
        )
        candidate_keep = root_keep.index_select(0, parent_indices)

        retained_indices = torch.nonzero(
            candidate_keep, as_tuple=False
        ).flatten().long().contiguous()
        filtered = points.index_select(0, retained_indices).contiguous()

        removed_root = roots & (~root_keep)
        removed_per_track = self._removed_components_per_track[:count]
        removed_per_track.zero_()
        removed_per_track.scatter_add_(
            0, track_ranks, removed_root.to(torch.int32)
        )
        instances_with_removed = (removed_per_track > 0).sum()
        statistics = torch.stack(
            (
                roots.sum(),
                root_sizes.max(),
                root_keep.sum(),
                removed_root.sum(),
                torch.where(removed_root, root_sizes, 0).sum(),
                track_count,
                instances_with_removed,
                track_count - instances_with_removed,
            )
        ).to(torch.int32)
        resolved_statistics = self._op_statistics[: len(_STATISTIC_NAMES)]
        resolved_statistics.copy_(statistics)

        retained_count = int(filtered.shape[0])
        return filtered, retained_indices, candidate_keep, {
            "outlier_filter_input_count": count,
            "outlier_filter_retained_count": retained_count,
            "outlier_filter_removed_count": count - retained_count,
            "_outlier_filter_statistics": resolved_statistics,
        }
