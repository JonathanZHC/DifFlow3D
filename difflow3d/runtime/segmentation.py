"""Per-object (segment) handling of one concatenated multi-object input.

The anchors handed to the model are object-major; every object is one segment. This module provides the three
per-object steps the plain single-cloud pipeline lacks:

* `allocate_targets`: the exact anchor count of every object for a total bucket (host ints; multiples of 16,
  a floor of 256/128/64/32 by object size, proportional to the candidate counts otherwise).
* `select_per_segment`: the deployed uniform-stride selection (or repeat) and Morton order applied inside each
  object, returning object-major anchors.
* `SegmentNormalization`: one canonical similarity transform per object (centroid -> model centre, RMS radius ->
  model radius), with the per-object re-anchor hysteresis of AnchoredNormalization, materialised as per-point
  centre / scale tensors that the runner stages with and inverts the outputs with.

The level-1 total of the model pyramid is N1 = sum(min(n_i, 1024)); the runner keys its CUDA graphs by (N0, N1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from difflow3d.model.encoder import INPUT_POINT_MULTIPLE, REFERENCE_POINTS
from difflow3d.model.segments import MIN_SEGMENT_POINTS

MAX_SEGMENTS = 16              # segment-table capacity of a captured graph (>= the tracker's total slot capacity)
FLOOR_TIERS = (256, 128, 64, MIN_SEGMENT_POINTS)   # per-object anchor floors, largest first


def _round16(value: int) -> int:
    return max(INPUT_POINT_MULTIPLE, (int(value) // INPUT_POINT_MULTIPLE) * INPUT_POINT_MULTIPLE)


def _floor_tier(count: int) -> int:
    """Largest floor tier an object's candidates fill without repetition (MIN_SEGMENT_POINTS below that)."""
    return next((tier for tier in FLOOR_TIERS if tier <= count), MIN_SEGMENT_POINTS)


def allocate_targets(counts: list[int], total: int) -> list[int]:
    """Anchor count per object (sum == total, multiples of INPUT_POINT_MULTIPLE).

    Floors: every object gets the largest tier of 256/128/64/32 its candidates fill (an object below 32 candidates
    is repeat-padded to 32); when the floors do not fit the bucket they step down the tiers (largest first). The
    rest of the bucket is shared in proportion to the candidates: repeat-padded when the whole frame has fewer
    candidates than the bucket (as the single-cloud path), otherwise water-filled up to each object's candidate
    count (rounded down to a multiple of 16)."""
    if len(counts) > MAX_SEGMENTS:
        raise ValueError(f"at most {MAX_SEGMENTS} objects per DifFlow input, got {len(counts)} (raise "
                         "difflow3d.runtime.segmentation.MAX_SEGMENTS above the tracker's total slot capacity)")
    total = int(total)
    if len(counts) == 1:
        return [total]
    if min_segment_total(len(counts)) > total:
        raise ValueError(f"bucket {total} too small for {len(counts)} objects (needs >= "
                         f"{min_segment_total(len(counts))})")
    caps = [max(MIN_SEGMENT_POINTS, (c // INPUT_POINT_MULTIPLE) * INPUT_POINT_MULTIPLE) for c in counts]
    if sum(caps) <= total:
        # Fewer candidates than the bucket: every object keeps all its points, the rest is repeat-padded in
        # proportion (like the single-cloud repeat).
        extra = total - sum(caps)
        share = [extra * c / max(sum(counts), 1) for c in counts]
        targets = [cap + _round16(s) if s >= INPUT_POINT_MULTIPLE else cap for cap, s in zip(caps, share)]
        targets[-1] += total - sum(targets)          # rounding remainder
        return targets
    floors = [_floor_tier(c) for c in counts]
    while sum(floors) > total:
        largest = max(floors)
        floors = [f // 2 if f == largest else f for f in floors]
    targets = list(floors)
    remaining = total - sum(targets)
    weights = [max(c - f, 0) for c, f in zip(counts, floors)]
    for _ in range(4):
        room = [cap - t for cap, t in zip(caps, targets)]
        active = [i for i, r in enumerate(room) if r > 0 and weights[i] > 0]
        if remaining < INPUT_POINT_MULTIPLE or not active:
            break
        wsum = sum(weights[i] for i in active)
        for i in active:
            give = min(room[i], (remaining * weights[i] // wsum) // INPUT_POINT_MULTIPLE * INPUT_POINT_MULTIPLE)
            targets[i] += give
        remaining = total - sum(targets)
    # Final remainder to the object with the most room (keeps multiples of 16 since total and targets are).
    if remaining:
        i = max(range(len(targets)), key=lambda j: caps[j] - targets[j])
        targets[i] += remaining
    return targets


def min_segment_total(num_segments: int) -> int:
    """Smallest anchor total that fits ``num_segments`` objects."""
    return max(1, int(num_segments)) * MIN_SEGMENT_POINTS


def level1_total(targets: list[int]) -> int:
    return sum(min(t, REFERENCE_POINTS) for t in targets)


def segment_bounds(point_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[N]`` object-major ids -> (unique ids in order, counts) on the device (one small sync by the caller)."""
    change = torch.ones_like(point_ids, dtype=torch.bool)
    change[1:] = point_ids[1:] != point_ids[:-1]
    starts = change.nonzero(as_tuple=True)[0]
    ends = torch.cat([starts[1:], torch.tensor([point_ids.shape[0]], device=point_ids.device)])
    return point_ids.index_select(0, starts), ends - starts


def select_per_segment(points: torch.Tensor, counts: list[int], targets: list[int], morton_cell: float | None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform-stride selection (repeat when short) and Morton order inside every object segment.

    points: object-major candidates [N,3]; counts / targets host ints per object. Returns (anchors [T,3], local
    indices into points [T]) object-major."""
    device = points.device
    pieces = []
    start = 0
    for count, target in zip(counts, targets):
        if count >= target:
            # rounded linspace over [0, count-1] (the deployed uniform selection), count == target is the identity
            pos = torch.arange(target, device=device, dtype=torch.long)
            idx = pos if count == target else torch.div(pos * (count - 1) + (target - 1) // 2, target - 1, rounding_mode="floor")
        else:
            idx = torch.arange(count, device=device, dtype=torch.long).repeat((target + count - 1) // count)[:target]
        idx = idx + start
        if morton_cell:
            seg_points = points.index_select(0, idx)
            idx = idx.index_select(0, morton_order(seg_points, morton_cell))
        pieces.append(idx)
        start += count
    local = torch.cat(pieces).contiguous()
    return points.index_select(0, local).contiguous(), local


def _spread_bits(value: torch.Tensor) -> torch.Tensor:
    value = value & 0x1FFFFF
    value = (value | (value << 32)) & 0x1F00000000FFFF
    value = (value | (value << 16)) & 0x1F0000FF0000FF
    value = (value | (value << 8)) & 0x100F00F00F00F00F
    value = (value | (value << 4)) & 0x10C30C30C30C30C3
    value = (value | (value << 2)) & 0x1249249249249249
    return value


def morton_order(points: torch.Tensor, cell: float) -> torch.Tensor:
    quantized = torch.floor((points - points.min(dim=0).values) / float(cell)).to(torch.int64).clamp_(0, (1 << 21) - 1)
    key = _spread_bits(quantized[:, 0]) | (_spread_bits(quantized[:, 1]) << 1) | (_spread_bits(quantized[:, 2]) << 2)
    return torch.argsort(key)


@dataclass
class SegmentTransform:
    """Per-object canonical transform: model = (world - center_j) * scale_j + model_center for object j. Shared by
    both frames of a pair (each frame expands it over its own object layout)."""
    centers: torch.Tensor         # [S,3]
    scales: torch.Tensor          # [S]
    model_center: torch.Tensor    # [3]
    per_object: tuple[tuple[tuple[float, float, float], float], ...]   # host (centroid, radius) per object

    def expand(self, counts) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-point (center [N,3], scale [N]) of an object-major frame with these per-object counts."""
        if len(counts) != self.scales.shape[0]:
            raise ValueError(f"transform has {self.scales.shape[0]} objects, layout {tuple(counts)}")
        seg = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(list(counts))).to(self.scales.device)
        return self.centers.index_select(0, seg), self.scales.index_select(0, seg)


class SegmentNormalization:
    """AnchoredNormalization per object: each object is centred and scaled on its own, re-anchored only when its
    RMS radius drifts by more than scale_ratio or its centroid by more than center_shift radii."""

    def __init__(self, model_center, model_radius: float, scale_ratio: float = 1.5, center_shift: float = 0.3,
                 restage_previous: bool = True) -> None:
        self.model_center = tuple(float(v) for v in model_center)
        self.model_radius = float(model_radius)
        self.scale_ratio = float(scale_ratio)
        self.center_shift = float(center_shift)
        self.restage_previous = bool(restage_previous)
        self.reanchor_count = 0
        self._anchors: dict[int, tuple[tuple[float, float, float], float]] = {}   # per object id
        self._model_center: torch.Tensor | None = None

    def reset(self) -> None:
        self._anchors.clear()

    def update(self, points: torch.Tensor, object_ids: list[int], counts: list[int]) -> tuple[SegmentTransform, bool]:
        """Per-object centroid / RMS radius (one device -> host copy of 4 x S values), hysteresis on the host."""
        device = points.device
        seg = torch.repeat_interleave(torch.arange(len(counts), device=device), torch.tensor(counts, device=device))
        sums = torch.zeros(len(counts), 3, device=device, dtype=points.dtype).index_add_(0, seg, points)
        n = torch.tensor(counts, device=device, dtype=points.dtype).unsqueeze(1)
        centers = sums / n
        d2 = ((points - centers.index_select(0, seg)) ** 2).sum(1)
        radii = (torch.zeros(len(counts), device=device, dtype=points.dtype).index_add_(0, seg, d2) / n.squeeze(1)).sqrt()
        stats = torch.cat([centers, radii.unsqueeze(1)], 1).tolist()
        changed = False
        per_object = []
        use_center, use_scale = [], []
        for oid, (cx, cy, cz, r) in zip(object_ids, stats):
            r = max(r, 1.0e-6)
            anchor = self._anchors.get(oid)
            if anchor is not None:
                (ax, ay, az), ar = anchor
                drift = math.dist((cx, cy, cz), (ax, ay, az))
                if max(r / ar, ar / r) <= self.scale_ratio and drift <= self.center_shift * ar:
                    per_object.append(anchor); use_center.append(anchor[0]); use_scale.append(self.model_radius / ar)
                    continue
                changed = True
                self.reanchor_count += 1
            self._anchors[oid] = ((cx, cy, cz), r)
            per_object.append(self._anchors[oid]); use_center.append((cx, cy, cz)); use_scale.append(self.model_radius / r)
        for oid in list(self._anchors):
            if oid not in object_ids:
                del self._anchors[oid]
        model_center = self._model_center
        if model_center is None or model_center.device != device or model_center.dtype != points.dtype:
            model_center = torch.tensor(self.model_center, device=device, dtype=points.dtype)
            self._model_center = model_center
        centers = torch.tensor(use_center, device=device, dtype=points.dtype)
        scales = torch.tensor(use_scale, device=device, dtype=points.dtype)
        return SegmentTransform(centers, scales, model_center, tuple(per_object)), changed
