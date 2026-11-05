"""Object segments of one concatenated point cloud, per pyramid level.

Several tracked objects are run through DifFlow3D as ONE cloud (one forward, one CUDA graph) whose points are
grouped object-major. Every neighbour search, FPS and feature matching stays inside the object's segment, so the
objects do not interact (as if each ran alone) while the batch stays a single fixed-shape graph.

A `Segments` object is built from the per-object input counts (host ints, fixed by the graph key) and holds, per
level l in 0..4, the segment id of every point `ids[l]` ([1, N_l] int32) and the FPS tables that sample level l+1
from level l: `offsets`, `counts` (level-l ranges), `out_offsets`, `npoints` (level-(l+1) ranges). All tensors
are allocated once per (counts, device) and reused, so a captured graph reads static buffers.

Per-object pyramid: level 1 = min(n, level1_points) points (the input itself when n <= level1_points), then
n1/2, n1/4, n1/16 (encoder.coarse_level_sizes scaled per object)."""
from __future__ import annotations

from functools import lru_cache

import torch

from .encoder import INPUT_POINT_MULTIPLE, REFERENCE_POINTS


MIN_SEGMENT_POINTS = 32      # smallest object segment: levels 32/16/8/2; kNN wider than a segment cycles its members


def level_sizes(n: int, level1_points: int) -> tuple[int, int, int, int, int]:
    """Per-object level sizes (N, L1, L2, L3, L4)."""
    if n < MIN_SEGMENT_POINTS or n % INPUT_POINT_MULTIPLE:
        raise ValueError(f"an object segment needs at least {MIN_SEGMENT_POINTS} points, a multiple of "
                         f"{INPUT_POINT_MULTIPLE}; got {n}")
    l1 = min(n, int(level1_points)) if n <= REFERENCE_POINTS else int(level1_points)
    return n, l1, l1 // 2, l1 // 4, l1 // 16


MAX_SEGMENTS = 16
LEVEL1_CAPACITY_STEP = 512   # multi-object graphs size level 1 in these steps (the rest is a padding segment)


def level1_capacity(n0: int, level1_real: int) -> int:
    """Level-1 size of the graph that runs an N0-point multi-object input whose objects need ``level1_real``
    level-1 points: N0 when N0 <= REFERENCE_POINTS (level 1 is the input), otherwise level1_real rounded up to
    LEVEL1_CAPACITY_STEP (at most N0). The quantisation bounds the graphs per bucket; the per-frame split of the
    bucket between objects would otherwise give a new N1 (and a new graph capture) almost every frame."""
    if n0 <= REFERENCE_POINTS:
        return int(n0)
    step = LEVEL1_CAPACITY_STEP
    return int(min(n0, -(-int(level1_real) // step) * step))


def level1_capacities(n0: int) -> tuple[int, ...]:
    """Every level-1 capacity an N0-point multi-object graph can have."""
    if n0 <= REFERENCE_POINTS:
        return (int(n0),)
    return tuple(sorted({level1_capacity(n0, n1) for n1 in range(REFERENCE_POINTS, n0 + 1, INPUT_POINT_MULTIPLE)}))


class Segments:
    """Static buffers describing the object layout of one graph shape (N0 points, level-1 capacity N1); `write()`
    refills them for any layout of N0 points whose objects fit N1, so a captured graph serves every split.

    Objects are segments 0..S-1. When the objects need fewer level-1 points than the capacity, the remainder is a
    padding segment (id S, no level-0 points): FPS samples it from the whole input, it has its own levels
    1..4 (P, P/2, P/4, P/16), and since every neighbour search is restricted to the query's segment id, no real
    point ever reads it (its level-1 grouping finds no level-0 member and reads point 0; its outputs are unused)."""

    def __init__(self, counts: tuple[int, ...], level1_points: int, device: torch.device,
                 level1_total: int | None = None) -> None:
        self.level1_points = int(level1_points)
        self.device = device
        i32 = dict(dtype=torch.int32, device=device)
        counts = tuple(int(c) for c in counts)
        real = self._level_counts(counts)
        n0, n1_real = sum(real[0]), sum(real[1])
        self.level1_total = int(level1_total) if level1_total is not None else n1_real
        if self.level1_total < n1_real or self.level1_total % INPUT_POINT_MULTIPLE:
            raise ValueError(f"level-1 capacity {self.level1_total} does not fit {counts} (needs {n1_real})")
        pad = self.level1_total - n1_real
        self.totals: tuple[int, ...] = (n0, self.level1_total, sum(real[2]) + pad // 2, sum(real[3]) + pad // 4,
                                        sum(real[4]) + pad // 16)
        # The identity level-1 shortcut (level 0 reused as level 1) is a graph property: every object of every
        # layout must be at most level1_points, i.e. N0 itself is.
        self.identity_l1 = n0 <= self.level1_points and self.level1_total == n0
        self.ids: list[torch.Tensor] = [torch.zeros((1, t), **i32) for t in self.totals]
        self._tables: list[torch.Tensor] = [torch.zeros((5, MAX_SEGMENTS + 1), **i32) for _ in range(4)]
        self.fps: list[tuple] = []
        self.write(counts)

    def _level_counts(self, counts):
        sizes = [level_sizes(c, self.level1_points) for c in counts]
        return tuple(tuple(s[l] for s in sizes) for l in range(5))

    def write(self, counts: tuple[int, ...]) -> None:
        """Refill the layout buffers for `counts` (same N0, objects fitting this graph's level-1 capacity)."""
        counts = tuple(int(c) for c in counts)
        if len(counts) > MAX_SEGMENTS:
            raise ValueError(f"at most {MAX_SEGMENTS} objects, got {len(counts)}")
        real = self._level_counts(counts)
        n0, n1_real = sum(real[0]), sum(real[1])
        if n0 != self.totals[0] or n1_real > self.level1_total:
            raise ValueError(f"layout {counts} (N0 {n0}, N1 {n1_real}) does not fit this graph "
                             f"(N0 {self.totals[0]}, N1 capacity {self.level1_total})")
        pad = self.level1_total - n1_real
        # level counts incl. the padding segment (0 level-0 points)
        pad_levels = (0, pad, pad // 2, pad // 4, pad // 16)
        level_counts = [list(lc) + ([pad_levels[l]] if pad else []) for l, lc in enumerate(real)]
        self.counts, self.level_counts = counts, tuple(tuple(lc[:len(counts)]) for lc in level_counts)
        S = len(level_counts[0])
        for l in range(5):
            seg = torch.repeat_interleave(torch.arange(S, dtype=torch.int32), torch.tensor(level_counts[l]))
            self.ids[l][0].copy_(seg.to(self.device, non_blocking=False))
        self.fps = []
        for l in range(4):
            lc, nxt = level_counts[l], level_counts[l + 1]
            rows = torch.zeros((5, MAX_SEGMENTS + 1), dtype=torch.int32)
            rows[0, :S] = torch.tensor([sum(lc[:i]) for i in range(S)])
            rows[1, :S] = torch.tensor(lc)
            rows[2, :S] = torch.tensor([sum(nxt[:i]) for i in range(S)])
            rows[3, :S] = torch.tensor(nxt)
            rows[4, :S] = torch.tensor([int(l == 0 and a == b) for a, b in zip(lc, nxt)])   # identity level 1
            if pad and l == 0:
                rows[0, S - 1], rows[1, S - 1] = 0, n0      # the padding segment samples the whole input
            self._tables[l].copy_(rows.to(self.device))
            t = self._tables[l]
            # (offsets, counts, out_offsets, npoints, identity, total_out, max_count, max_npoint); the host ints
            # are graph properties (maxima over all layouts of this shape: the level totals bound them).
            self.fps.append((t[0], t[1], t[2], t[3], t[4], self.totals[l + 1], self.totals[l], self.totals[l + 1]))

    @property
    def num_segments(self) -> int:
        return len(self.counts)

    def __repr__(self) -> str:
        return f"Segments(counts={self.counts}, totals={self.totals})"


@lru_cache(maxsize=64)
def _cached(counts: tuple[int, ...], level1_points: int, device_str: str, level1_total) -> Segments:
    return Segments(counts, level1_points, torch.device(device_str), level1_total)


def segments_for(counts: tuple[int, ...], level1_points: int, device: torch.device,
                 level1_total: int | None = None) -> Segments:
    """One `Segments` per (counts, level1_points, device, capacity), cached (its tensors are graph-static)."""
    return _cached(tuple(int(c) for c in counts), int(level1_points), str(device), level1_total)
