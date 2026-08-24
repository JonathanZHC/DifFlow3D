#!/usr/bin/env python3
"""CUDA checks for the track-aware relative-size voxel filter."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from difflow3d.ops.pointnet2 import pointnet2_utils
from difflow3d.runtime.preprocessing import AdaptivePointPreprocessor
from difflow3d.runtime.voxel_outlier import (
    CudaVoxelComponentOutlierFilter,
    resolve_voxel_outlier_statistics,
)


def _line(start: int, count: int, *, y: int = 0) -> list[tuple[int, int, int]]:
    return [(start + index, y, 0) for index in range(count)]


def _metadata(
    coordinates: list[tuple[int, int, int]],
    track_ids: list[int] | None,
):
    points = torch.tensor(coordinates, dtype=torch.float32, device="cuda") + 0.1
    ids = (
        torch.tensor(track_ids, dtype=torch.int32, device="cuda")
        if track_ids is not None
        else None
    )
    return AdaptivePointPreprocessor._voxel_downsample(points, 1.0, ids)


def _new_filter() -> CudaVoxelComponentOutlierFilter:
    return CudaVoxelComponentOutlierFilter(min_component_size_ratio=0.25)


def _apply(filter_, metadata):
    points, _, coords, absolute, keys, extents, input_map, ids = metadata
    filtered, _, candidate_keep, info = filter_.filter(
        points, coords, absolute, keys, extents, ids
    )
    resolve_voxel_outlier_statistics(info)
    return filtered, candidate_keep, input_map, info


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")
    if not pointnet2_utils.has_voxel_component_filter_op():
        raise RuntimeError(
            "Rebuild the extension first: bash scripts/build_pointnet2_ops.sh"
        )

    # [8, 4, 1]: threshold is 8 * 0.25 = 2, so 8 and 4 remain.
    coordinates = _line(0, 8) + _line(20, 4) + _line(40, 1)
    filtered, _, _, info = _apply(
        _new_filter(), _metadata(coordinates, [7] * len(coordinates))
    )
    assert len(filtered) == 12
    assert info["outlier_filter_statistics"]["retained_component_count"] == 2
    print("PASS relative-size threshold")

    # The comparison is intentionally strict: size == threshold is removed.
    coordinates = _line(0, 8) + _line(20, 2) + _line(40, 1)
    filtered, _, _, _ = _apply(
        _new_filter(), _metadata(coordinates, [3] * len(coordinates))
    )
    assert len(filtered) == 8
    print("PASS strict threshold boundary")

    # A single component is always retained for any configured ratio below one.
    coordinates = _line(0, 8)
    filtered, _, _, info = _apply(
        _new_filter(), _metadata(coordinates, [4] * len(coordinates))
    )
    assert len(filtered) == 8
    assert (
        info["outlier_filter_statistics"][
            "instances_all_components_retained"
        ]
        == 1
    )
    print("PASS largest-only instance")

    # Two tracks occupying identical voxels remain separate instances.
    overlap = _line(0, 3)
    filtered, _, _, info = _apply(
        _new_filter(), _metadata(overlap + overlap, [1] * 3 + [2] * 3)
    )
    assert len(filtered) == 6
    statistics = info["outlier_filter_statistics"]
    assert statistics["component_count"] == 2
    assert statistics["instance_count"] == 2
    print("PASS per-track components")

    # Dense duplicates represented by a rejected voxel are rejected together.
    dense = [(0, 0, 0)] * 3 + _line(1, 7) + [(20, 0, 0)] * 4
    filtered, candidate_keep, input_map, _ = _apply(
        _new_filter(), _metadata(dense, None)
    )
    input_keep = candidate_keep.index_select(0, input_map)
    assert len(filtered) == 8
    assert input_keep[:10].all()
    assert not input_keep[10:].any()
    print("PASS dense input keep mask and virtual track")


if __name__ == "__main__":
    main()
