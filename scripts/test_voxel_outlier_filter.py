#!/usr/bin/env python3
"""Correctness checks for the CUDA spatial/temporal voxel filter."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from difflow3d.ops.pointnet2 import pointnet2_utils


def _sorted_voxels(
    coordinates: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    absolute = torch.tensor(coordinates, dtype=torch.int32, device="cuda")
    order = sorted(range(len(coordinates)), key=coordinates.__getitem__)
    order_tensor = torch.tensor(order, dtype=torch.long, device="cuda")
    absolute = absolute.index_select(0, order_tensor).contiguous()
    shifted64 = absolute.to(torch.int64) - absolute.to(torch.int64).amin(dim=0)
    extents64 = shifted64.amax(dim=0) + 1
    keys = (
        shifted64[:, 0] * (extents64[1] * extents64[2])
        + shifted64[:, 1] * extents64[2]
        + shifted64[:, 2]
    )
    return (
        keys.contiguous(),
        shifted64.to(torch.int32).contiguous(),
        absolute,
        extents64.to(torch.int32).contiguous(),
    )


def _components(
    coordinates: list[tuple[int, int, int]],
) -> list[set[tuple[int, int, int]]]:
    remaining = set(coordinates)
    result: list[set[tuple[int, int, int]]] = []
    while remaining:
        component: set[tuple[int, int, int]] = set()
        queue = deque([remaining.pop()])
        while queue:
            voxel = queue.popleft()
            component.add(voxel)
            x, y, z = voxel
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = (x + dx, y + dy, z + dz)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            queue.append(neighbor)
        result.append(component)
    return result


def _reference_mask(
    coordinates: list[tuple[int, int, int]],
    previous: list[tuple[int, int, int]],
    *,
    tiny: int,
    max_small_fraction: float,
    radius: int,
    min_supported_fraction: float,
) -> list[bool]:
    components = _components(coordinates)
    largest = max(map(len, components))
    max_small = max(tiny, math.ceil(max_small_fraction * len(coordinates)))
    previous_set = set(previous)
    keep_by_voxel: dict[tuple[int, int, int], bool] = {}
    for component in components:
        size = len(component)
        if size == largest:
            keep = True
        elif size <= tiny:
            keep = False
        elif size <= max_small and previous:
            supported = sum(
                any(
                    (x + dx, y + dy, z + dz) in previous_set
                    for dx in range(-radius, radius + 1)
                    for dy in range(-radius, radius + 1)
                    for dz in range(-radius, radius + 1)
                )
                for x, y, z in component
            )
            keep = supported >= min_supported_fraction * size
        else:
            keep = True
        keep_by_voxel.update((voxel, keep) for voxel in component)
    return [keep_by_voxel[voxel] for voxel in coordinates]


def _run_case(
    name: str,
    coordinates: list[tuple[int, int, int]],
    previous: list[tuple[int, int, int]],
    *,
    tiny: int = 2,
    max_small_fraction: float = 0.6,
    radius: int = 1,
    min_supported_fraction: float = 0.3,
) -> None:
    keys, shifted, absolute, extents = _sorted_voxels(coordinates)
    if previous:
        previous_tensor = torch.tensor(
            sorted(previous), dtype=torch.int32, device="cuda"
        ).contiguous()
    else:
        previous_tensor = torch.empty((0, 3), dtype=torch.int32, device="cuda")
    count = len(coordinates)
    parents = torch.empty(count, dtype=torch.int32, device="cuda")
    sizes = torch.empty_like(parents)
    supported = torch.empty_like(parents)
    largest = torch.empty(1, dtype=torch.int32, device="cuda")
    keep = torch.empty(count, dtype=torch.bool, device="cuda")
    statistics = torch.empty(11, dtype=torch.int32, device="cuda")
    actual = pointnet2_utils.voxel_component_keep_mask(
        keys,
        shifted,
        absolute,
        extents,
        previous_tensor,
        tiny_component_max_voxels=tiny,
        max_small_component_voxels=max(
            tiny, math.ceil(max_small_fraction * count)
        ),
        support_radius_voxels=radius,
        min_supported_fraction=min_supported_fraction,
        parents=parents,
        component_sizes=sizes,
        supported_counts=supported,
        largest_component_size=largest,
        keep_mask=keep,
        statistics=statistics,
    )

    sorted_coordinates = [tuple(map(int, row)) for row in absolute.cpu()]
    expected = _reference_mask(
        sorted_coordinates,
        previous,
        tiny=tiny,
        max_small_fraction=max_small_fraction,
        radius=radius,
        min_supported_fraction=min_supported_fraction,
    )
    actual_list = actual.cpu().tolist()
    statistics_list = statistics.cpu().tolist()
    if actual_list != expected:
        raise AssertionError(
            f"{name}: CUDA mask {actual_list} != reference {expected}; "
            f"statistics={statistics_list}"
        )
    components = _components(sorted_coordinates)
    if statistics_list[0] != len(components):
        raise AssertionError(f"{name}: incorrect component count statistics")
    if statistics_list[1] != max(map(len, components)):
        raise AssertionError(f"{name}: incorrect largest-component statistics")
    if statistics_list[7] != count - sum(actual_list):
        raise AssertionError(f"{name}: incorrect removed-voxel statistics")
    print(f"PASS {name}: retained {sum(actual_list)}/{count} voxels")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")
    if not pointnet2_utils.has_voxel_component_filter_op():
        raise RuntimeError(
            "Rebuild the extension first: bash scripts/build_pointnet2_ops.sh"
        )

    large = [(x, 0, 0) for x in range(5)]
    medium = [(20 + x, 0, 0) for x in range(3)]
    tiny = [(40, 0, 0)]
    current = large + medium + tiny

    _run_case("first-frame-medium-kept", current, [])
    _run_case("unsupported-medium-rejected", current, large)
    _run_case("support-rate-keeps-component", current, large + [medium[0]])
    _run_case(
        "radius-one-support",
        current,
        large + [(x, 1, 0) for x, _, _ in medium],
    )
    _run_case(
        "largest-ties-always-kept",
        [(0, 0, 0), (10, 0, 0), (20, 0, 0)],
        [],
    )


if __name__ == "__main__":
    main()
