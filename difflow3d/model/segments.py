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

import numpy as np
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
    padding segment (id S, no level-0 points): a prefix copy of the first input points (never sampled), with its
    own levels 1..4 (P, P/2, P/4, P/16); since every neighbour search is restricted to the query's segment id, no
    real point ever reads it (its level-1 grouping finds no level-0 member and reads point 0; its outputs are
    unused)."""

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
        # All per-level ids and the 4 FPS tables live in ONE device buffer (graph-static views), refilled per frame
        # with one pinned host->device copy (no stream synchronisation).
        width = MAX_SEGMENTS + 1
        self._table_size = 5 * width
        self._flat = torch.zeros(sum(self.totals) + 4 * self._table_size, **i32)
        views, offset = [], 0
        for t in self.totals:
            views.append(self._flat[offset:offset + t].view(1, t)); offset += t
        self.ids: list[torch.Tensor] = views
        self._tables: list[torch.Tensor] = []
        for _ in range(4):
            self._tables.append(self._flat[offset:offset + self._table_size].view(5, width)); offset += self._table_size
        self.fps: list[tuple] = []
        for l in range(4):
            t = self._tables[l]
            # (offsets, counts, out_offsets, npoints, identity, total_out, max_count, max_npoint); the host ints
            # are graph properties valid for every layout of this shape. max_count bounds the size of a SAMPLED
            # segment (FPS vector width): level 0 only samples objects above level1_points (<= N0); from level 1 on
            # an object has at most level_sizes(level1_points)[l] points (the padding segment is never sampled).
            bound = self.totals[0] if l == 0 else min(self.totals[l], level_sizes(self.level1_points, self.level1_points)[l])
            self.fps.append((t[0], t[1], t[2], t[3], t[4], self.totals[l + 1], bound, self.totals[l + 1]))
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
        pad_levels = (0, pad, pad // 2, pad // 4, pad // 16)
        # level counts incl. the padding segment (0 level-0 points)
        level_counts = [list(lc) + ([pad_levels[l]] if pad else []) for l, lc in enumerate(real)]
        self.counts, self.level_counts = counts, tuple(tuple(lc[:len(counts)]) for lc in level_counts)
        S = len(level_counts[0])
        width = MAX_SEGMENTS + 1
        parts = [np.repeat(np.arange(S, dtype=np.int32), level_counts[l]) for l in range(5)]
        for l in range(4):
            lc, nxt = np.asarray(level_counts[l]), np.asarray(level_counts[l + 1])
            rows = np.zeros((5, width), dtype=np.int32)
            rows[0, :S] = np.concatenate(([0], np.cumsum(lc)[:-1]))
            rows[1, :S] = lc
            rows[2, :S] = np.concatenate(([0], np.cumsum(nxt)[:-1]))
            rows[3, :S] = nxt
            rows[4, :S] = (lc == nxt) if l == 0 else 0                 # identity level 1
            if pad:
                rows[4, S - 1] = 1                                     # the padding segment is a prefix copy, never sampled
                if l == 0:
                    rows[0, S - 1], rows[1, S - 1] = 0, n0             # ... of the first input points
            parts.append(rows.reshape(-1))
        host = torch.from_numpy(np.concatenate(parts))
        if self._flat.is_cuda:
            host = host.pin_memory()
        self._flat.copy_(host, non_blocking=True)

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
