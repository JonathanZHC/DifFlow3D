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


def to_device(values, device: torch.device, dtype=torch.int64) -> torch.Tensor:
    """Small host list/array -> device tensor through pinned memory, without a stream synchronisation (a pageable
    host->device copy waits for all queued GPU work, which stalls the whole pipeline)."""
    host = torch.as_tensor(values, dtype=dtype)
    if device.type != "cuda":
        return host.to(device)
    return host.pin_memory().to(device, non_blocking=True)


@dataclass(frozen=True)
class SegmentSummary:
    """Per-object facts of one object-major candidate cloud, read back in ONE device->host copy."""
    ids: tuple[int, ...]                                # object (track) id per segment, in order
    counts: tuple[int, ...]                             # candidates per object
    centroids: tuple[tuple[float, float, float], ...]   # world centroid of each object's candidates
    radii: tuple[float, ...]                            # RMS radius about that centroid


def summarize_segments(points: torch.Tensor, point_ids: torch.Tensor) -> SegmentSummary:
    """Segments of contiguous equal ids (object-major) with their count, centroid and RMS radius: one reduction
    into a fixed [MAX_SEGMENTS, 6] float64 table and one host read (the only sync of the multi-object path; the
    single-cloud path pays the same one in voxel-2)."""
    change = torch.ones_like(point_ids, dtype=torch.int64)
    change[1:] = (point_ids[1:] != point_ids[:-1]).to(torch.int64)
    seg = torch.cumsum(change, 0) - 1
    p = points.to(torch.float64)
    rows = torch.cat([torch.ones_like(p[:, :1]), p, (p * p).sum(1, keepdim=True), point_ids.to(torch.float64)[:, None]], 1)
    table = torch.zeros((MAX_SEGMENTS + 1, 6), dtype=torch.float64, device=points.device)
    table.index_add_(0, seg.clamp_max(MAX_SEGMENTS), rows)
    host = table.cpu().tolist()
    used = [row for row in host if row[0] > 0]
    if len(used) > MAX_SEGMENTS:
        raise ValueError(f"more than {MAX_SEGMENTS} objects per DifFlow input (raise "
                         "difflow3d.runtime.segmentation.MAX_SEGMENTS above the tracker's total slot capacity)")
    ids, counts, centroids, radii = [], [], [], []
    for n, sx, sy, sz, sq, sid in used:
        c = (sx / n, sy / n, sz / n)
        ids.append(int(round(sid / n))); counts.append(int(n)); centroids.append(c)
        radii.append(math.sqrt(max(sq / n - (c[0] ** 2 + c[1] ** 2 + c[2] ** 2), 0.0)))
    return SegmentSummary(tuple(ids), tuple(counts), tuple(centroids), tuple(radii))


def select_per_segment(points: torch.Tensor, counts: list[int], targets: list[int], morton_cell: float | None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform-stride selection (repeat when short) and Morton order inside every object segment, vectorised over
    the objects (a fixed handful of kernels, no host sync).

    points: object-major candidates [N,3]; counts / targets host ints per object. Returns (anchors [T,3], local
    indices into points [T], segment index per anchor [T]) object-major."""
    device = points.device
    total = int(sum(targets))
    starts = [0]
    for c in counts[:-1]:
        starts.append(starts[-1] + int(c))
    out_starts = [0]
    for t in targets[:-1]:
        out_starts.append(out_starts[-1] + int(t))
    params = to_device([list(counts), list(targets), starts, out_starts], device)       # [4, S]
    seg = torch.repeat_interleave(torch.arange(len(counts), device=device), params[1], output_size=total)
    count, target, start = params[0][seg], params[1][seg], params[2][seg]
    pos = torch.arange(total, device=device) - params[3][seg]
    # rounded linspace over [0, count-1] (the deployed uniform selection; count == target is the identity), or a
    # cyclic repeat when the object has fewer candidates than anchors
    stride = torch.div(pos * (count - 1) + torch.div(target - 1, 2, rounding_mode="floor"),
                       (target - 1).clamp_min(1), rounding_mode="floor")
    local = torch.where(count >= target, stride, torch.remainder(pos, count)) + start
    if morton_cell:
        selected = points.index_select(0, local)
        lo = torch.full((len(counts), 3), float("inf"), device=device, dtype=points.dtype)
        lo = lo.scatter_reduce(0, seg[:, None].expand(-1, 3), selected, reduce="amin", include_self=True)
        quantized = torch.floor((selected - lo[seg]) / float(morton_cell)).to(torch.int64).clamp_(0, (1 << 16) - 1)
        key = (_spread_bits(quantized[:, 0]) | (_spread_bits(quantized[:, 1]) << 1)
               | (_spread_bits(quantized[:, 2]) << 2) | (seg << 50))       # 48 Morton bits below the segment index
        order = torch.argsort(key)
        local, seg = local.index_select(0, order), seg.index_select(0, order)
    local = local.contiguous()
    return points.index_select(0, local).contiguous(), local, seg


def _spread_bits(value: torch.Tensor) -> torch.Tensor:
    value = value & 0x1FFFFF
    value = (value | (value << 32)) & 0x1F00000000FFFF
    value = (value | (value << 16)) & 0x1F0000FF0000FF
    value = (value | (value << 8)) & 0x100F00F00F00F00F
    value = (value | (value << 4)) & 0x10C30C30C30C30C3
    value = (value | (value << 2)) & 0x1249249249249249
    return value


@dataclass
class SegmentTransform:
    """Per-object canonical transform: model = (world - center_j) * scale_j + model_center for object j. Shared by
    both frames of a pair (each frame expands it over its own object layout)."""
    centers: torch.Tensor         # [S,3]
    scales: torch.Tensor          # [S]
    model_center: torch.Tensor    # [3]
    per_object: tuple[tuple[tuple[float, float, float], float], ...]   # host (centroid, radius) per object

    def expand(self, segment_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-point (center [N,3], scale [N]) of an object-major frame from its per-point segment index."""
        return self.centers.index_select(0, segment_index), self.scales.index_select(0, segment_index)


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

    def update(self, summary: "SegmentSummary", device: torch.device, dtype=torch.float32
               ) -> tuple[SegmentTransform, bool]:
        """Per-object hysteresis on the host from the candidates' centroid / RMS radius (SegmentSummary: already
        read back with the counts); one pinned host->device copy of the S centres and scales."""
        changed = False
        per_object = []
        use = []
        for oid, center, r in zip(summary.ids, summary.centroids, summary.radii):
            r = max(r, 1.0e-6)
            anchor = self._anchors.get(oid)
            if anchor is not None:
                ac, ar = anchor
                if max(r / ar, ar / r) <= self.scale_ratio and math.dist(center, ac) <= self.center_shift * ar:
                    per_object.append(anchor); use.append((*ac, self.model_radius / ar))
                    continue
                changed = True
                self.reanchor_count += 1
            self._anchors[oid] = (tuple(center), r)
            per_object.append(self._anchors[oid]); use.append((*center, self.model_radius / r))
        for oid in list(self._anchors):
            if oid not in summary.ids:
                del self._anchors[oid]
        model_center = self._model_center
        if model_center is None or model_center.device != device or model_center.dtype != dtype:
            model_center = torch.tensor(self.model_center, device=device, dtype=dtype)
            self._model_center = model_center
        packed = to_device(use, device, dtype)                                    # [S, 4]
        return SegmentTransform(packed[:, :3], packed[:, 3], model_center, tuple(per_object)), changed
