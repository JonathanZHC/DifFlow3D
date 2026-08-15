#!/usr/bin/env python3
"""Validate fused track-aware local recovery against the old per-track path."""

from __future__ import annotations

import torch

from difflow3d.runtime import SoftmaxAnchorMotionRecoverer


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    torch.manual_seed(7)

    # Three spatially overlapping tracks make cross-track leakage easy to catch.
    track_count = 3
    anchors_per_track = 256
    queries_per_track = 4096
    sigma_m = 0.025

    anchor_tracks = torch.arange(
        track_count, device=device, dtype=torch.int32
    ).repeat_interleave(anchors_per_track)
    query_tracks = torch.arange(
        track_count, device=device, dtype=torch.int32
    ).repeat_interleave(queries_per_track)

    anchors = 0.03 * torch.randn(
        track_count * anchors_per_track, 3, device=device
    )
    queries = 0.03 * torch.randn(
        track_count * queries_per_track, 3, device=device
    )

    # Give each track a distinct flow so any identity mixing is obvious.
    base_flow = torch.tensor(
        [[0.01, 0.00, 0.00], [0.00, 0.02, 0.00], [0.00, 0.00, -0.03]],
        device=device,
        dtype=torch.float32,
    )
    anchor_flow = base_flow.index_select(0, anchor_tracks.long()).contiguous()

    recoverer = SoftmaxAnchorMotionRecoverer(
        chunk_size=4096,
        softmax_sigma_m=sigma_m,
        backend="local",
        local_radius_sigma=4.0,
        local_hash_size_factor=4.0,
    )

    fused = recoverer.recover(
        query_points=queries,
        anchor_points=anchors,
        anchor_flow=anchor_flow,
        query_track_ids=query_tracks,
        anchor_track_ids=anchor_tracks,
        dt_s=1.0,
    ).flow

    reference = torch.empty_like(fused)
    for track_id in range(track_count):
        qmask = query_tracks == track_id
        amask = anchor_tracks == track_id
        reference[qmask] = recoverer.recover(
            query_points=queries[qmask],
            anchor_points=anchors[amask],
            anchor_flow=anchor_flow[amask],
            dt_s=1.0,
        ).flow

    torch.cuda.synchronize()
    max_error = (fused - reference).abs().max().item()
    print(f"max |fused-reference| = {max_error:.9e}")
    if max_error > 2.0e-6:
        raise RuntimeError("Track-aware fused recovery does not match per-track reference")
    print("Track-aware recovery test OK")


if __name__ == "__main__":
    main()
