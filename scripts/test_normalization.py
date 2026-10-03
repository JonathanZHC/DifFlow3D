#!/usr/bin/env python3
"""Canonical normalisation in the streaming runner (synthetic clouds, no data needed).

Inside the perceptive-safety-filter container:
    PYTHONPATH=/workspace/external/ScenePredictor/DifFlow3D /opt/tracking-venv/bin/python scripts/test_normalization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from test_variable_points import build_model, sphere  # noqa: E402
from difflow3d.runtime.normalization import AnchoredNormalization  # noqa: E402
from difflow3d.runtime.runner import DifFlow3DStreamingCudaGraphRunner  # noqa: E402

BUCKETS = (1024, 2048)


def runner(model, **kw):
    return DifFlow3DStreamingCudaGraphRunner(
        model, batch_size=1, num_points=2048, uncertainty=0.1, warmup=2, dt_s=1.0, second_base_voxel_size_m=0.01,
        second_candidate_ratio=1.1, auto_spatial_scale=False, fixed_spatial_scale=1.0, final_selection="uniform",
        point_buckets=BUCKETS, sort_anchors_morton=True, normalization=AnchoredNormalization(**kw))


def main() -> None:
    model = build_model()
    cloud = sphere(1500, 0.3, (0.2, -0.1, 0.9), seed=0)

    # 1) world round trip and a shared transform per pair
    r = runner(model)
    r.stage_world(cloud); r.replay_next()
    r.stage_world(cloud + 0.002); r.replay_next()
    src = r.source_points_world()[0]
    sel = r.source_selection_indices()
    assert torch.allclose(src, cloud[sel], atol=2e-5), float((src - cloud[sel]).abs().max())
    assert r._slot_transform[r._last_source_slot] is r._slot_transform[r._last_target_slot]
    flow = r.flow_world()[0]
    assert torch.isfinite(flow).all() and float(flow.norm(dim=1).mean()) < 0.05
    print("[ok] world round trip, one transform per pair")

    # 2) hysteresis: a slow drift inside the tolerance keeps the anchor, a jump re-anchors and re-encodes
    r = runner(model, scale_ratio=1.5, center_shift=0.3)
    for i in range(5):
        r.stage_world(cloud + torch.tensor([0.004 * i, 0.0, 0.0], device="cuda")); r.replay_next()
    assert r.normalization.reanchor_count == 0
    before = r.reencode_count
    r.stage_world(cloud + torch.tensor([0.3, 0.0, 0.0], device="cuda")); r.replay_next()
    assert r.normalization.reanchor_count == 1 and r.reencode_count == before + 1
    assert r._slot_transform[r._last_source_slot] is r._slot_transform[r._last_target_slot]
    print("[ok] hysteresis: drift kept, jump re-anchored with one re-encode")

    # 3) reset starts a new anchor without counting a re-anchor
    r.reset(); r.stage_world(cloud); r.replay_next()
    assert r.normalization.reanchor_count == 1
    print("[ok] reset")

    # 4) normalisation and auto spatial scale are exclusive
    try:
        DifFlow3DStreamingCudaGraphRunner(model, num_points=1024, auto_spatial_scale=True, normalization=AnchoredNormalization())
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("ALL NORMALIZATION TESTS PASSED")


if __name__ == "__main__":
    main()
