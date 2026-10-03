#!/usr/bin/env python3
"""Tests for bucketed exact-count preprocessing, Morton ordering, the dense-input
model shortcuts and the bucketed streaming runner.

Inside the perceptive-safety-filter container:
    PYTHONPATH=/workspace/external/ScenePredictor/DifFlow3D-dense \
      /opt/tracking-venv/bin/python scripts/test_variable_points.py [--human /tmp/real]

--human evaluates the whole runner (world-space API) on the recorded tracked-human
pairs with pseudo ground truth (gap 1) and prints the EPE by true-motion bin.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from difflow3d.model.difflow import PointConvBidirection  # noqa: E402
from difflow3d.runtime.checkpoint import load_checkpoint  # noqa: E402
from difflow3d.runtime.preprocessing import AdaptivePointPreprocessor  # noqa: E402
from difflow3d.runtime.runner import (  # noqa: E402
    DifFlow3DStreamingCudaGraphRunner,
    configure_fast_inference,
)

BUCKETS = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
CHECKPOINT = REPO_ROOT / "checkpoints" / "model_difflow_355_0.0114.pth"


def sphere(count: int, radius: float, center, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(count, 3, generator=g)
    v = v / v.norm(dim=1, keepdim=True) * radius
    return (v + torch.tensor(center)).cuda()


def build_model(fast_top: int = 0, hier: int = 0) -> PointConvBidirection:
    configure_fast_inference(True)
    model = PointConvBidirection(iters=4, coarse_iters=4, middle_iters=2, fine_iters=2)
    load_checkpoint(model, CHECKPOINT, strict=True)
    model.cuda().eval()
    for layer in model.modules():
        if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            layer.track_running_stats = False
    model.configure_dense_points(fast_top_level_min_points=fast_top, hier_cosine_min_points=hier)
    return model


def make_runner(model, num_points, buckets, *, morton, auto_scale=False, profiling=False):
    return DifFlow3DStreamingCudaGraphRunner(
        model,
        batch_size=1,
        num_points=num_points,
        uncertainty=0.1,
        warmup=3,
        dt_s=1.0,
        second_base_voxel_size_m=0.01,
        second_candidate_ratio=1.1,
        auto_spatial_scale=auto_scale,
        target_model_volume=5.0,
        fixed_spatial_scale=1.0,
        final_selection="uniform",
        enable_profiling=profiling,
        point_buckets=buckets,
        sort_anchors_morton=morton,
    )


def test_preprocessor() -> None:
    prep = AdaptivePointPreprocessor(
        num_points=8192, second_base_voxel_size_m=0.01, second_candidate_ratio=1.1,
        auto_spatial_scale=False, target_model_volume=1.0, final_selection="uniform",
        point_buckets=BUCKETS, sort_anchors_morton=True,
    )
    expected = {900: 1024, 1024: 1024, 1500: 1024, 3000: 2048, 6300: 6144, 9000: 8192}
    for count, bucket in expected.items():
        frame = sphere(count, 0.5, (0.0, 0.0, 0.0), seed=count)
        ids = torch.full((count,), 7, dtype=torch.int32, device="cuda")
        out = prep.prepare(frame, ids)
        assert out.world_points.shape == (bucket, 3), (count, out.world_points.shape)
        assert int(out.info["bucket"]) == bucket and int(out.info["anchor_count"]) == bucket
        assert out.point_ids is not None and out.point_ids.shape == (bucket,)
        sel = out.selection_indices
        if count >= 1024:
            assert sel.unique().numel() == bucket, "no duplicates expected above 1024 points"
        else:
            assert out.info["selection_mode"].endswith("repeat")
        # anchors are the selected frame points, in Morton order
        assert torch.allclose(frame[sel], out.world_points)
        key = prep._morton_order(out.world_points)
        assert torch.equal(key, torch.arange(bucket, device="cuda")), "anchors not Morton-sorted"
        if count > 1024:
            smaller = prep.refinalize(out, 1024)
            assert smaller.world_points.shape == (1024, 3)
            assert torch.allclose(frame[smaller.selection_indices], smaller.world_points)
    # voxel-2 path (above the candidate threshold)
    frame = sphere(12000, 0.5, (0.0, 0.0, 0.0), seed=1)
    out = prep.prepare(frame, collect_diagnostics=True)
    assert out.info["second_voxel_used"] and out.world_points.shape[0] in BUCKETS
    print(f"[ok] preprocessor buckets, Morton order, refinalize (voxel-2 case -> {out.world_points.shape[0]} anchors)")


@torch.inference_mode()
def test_runner_equivalence(model) -> None:
    """A multi-bucket runner reproduces the single-size runner at an exact bucket size."""
    frames = [sphere(4096, 0.5, (0.02 * k, 0.0, 0.0), seed=10 + k) for k in range(3)]
    single = make_runner(model, 4096, (4096,), morton=False)
    multi = make_runner(model, 8192, BUCKETS, morton=False)
    flows_single, flows_multi = [], []
    for frame in frames:
        single.stage_world(frame)
        if single.replay_next() is not None:
            flows_single.append(single.flow_world()[0].clone())
    for frame in frames:
        multi.stage_world(frame)
        if multi.replay_next() is not None:
            flows_multi.append(multi.flow_world()[0].clone())
            assert multi.current_point_count() == 4096
    # The diffusion head draws noise inside the graphs (torch.randn), so two
    # runners never agree bit-for-bit; compare their errors on the known motion.
    truth = torch.tensor([0.02, 0.0, 0.0], device="cuda")
    for a, b in zip(flows_single, flows_multi):
        epe_a = (a - truth).norm(dim=1).mean().item()
        epe_b = (b - truth).norm(dim=1).mean().item()
        assert epe_a < 0.02 and epe_b < 0.02, (epe_a, epe_b)
        assert abs(epe_a - epe_b) < 0.003, f"single {epe_a:.4f} vs multi {epe_b:.4f}"
    del single, multi
    torch.cuda.empty_cache()
    print(f"[ok] multi-bucket runner matches single-size runner at 4096 (EPE {epe_a*1e3:.1f} vs {epe_b*1e3:.1f} mm)")


@torch.inference_mode()
def test_runner_streaming(model) -> None:
    """Varying point counts: bucket switching, re-encode count, output sizes, motion recovery."""
    counts = [6300, 6300, 5900, 7000, 900, 1500, 6300, 6300]
    expected_pairs = [6144, 5120, 5120, 1024, 1024, 1024, 6144]   # min of the two natural buckets
    expected_reencode = [0, 1, 0, 1, 0, 0, 1]                       # previous frame re-encoded
    runner = make_runner(model, 8192, BUCKETS, morton=True)
    step = torch.tensor([0.03, 0.0, 0.0], device="cuda")
    previous = None
    pairs = 0
    for k, count in enumerate(counts):
        frame = sphere(count, 0.5, (0.0, 0.0, 0.0), seed=100 + k) + step * k
        runner.stage_world(frame)
        out = runner.replay_next()
        if out is None:
            continue
        assert runner.current_point_count() == expected_pairs[pairs], (pairs, runner.current_point_count())
        assert runner.reencode_count == sum(expected_reencode[: pairs + 1]), (pairs, runner.reencode_count)
        flow = runner.flow_world()[0]
        src = runner.source_points_world()[0]
        assert flow.shape == src.shape == (expected_pairs[pairs], 3)
        assert runner.source_selection_indices().shape[0] == expected_pairs[pairs]
        epe = (flow - step).norm(dim=1).mean().item()
        assert epe < 0.02, f"pair {pairs}: EPE {epe:.4f} m for a 3 cm translation"
        pairs += 1
    assert pairs == len(expected_pairs)
    del runner
    torch.cuda.empty_cache()
    print(f"[ok] streaming over {counts}: pair buckets {expected_pairs}, re-encodes {runner_reencodes(expected_reencode)}")


def runner_reencodes(flags):
    return sum(flags)


@torch.inference_mode()
def timing(model_plain, model_dense) -> None:
    """Graph-replay time per bucket, dense shortcuts off/on (synthetic sphere, after warm-up)."""
    for name, model in (("plain", model_plain), ("dense-shortcuts", model_dense)):
        runner = make_runner(model, 8192, BUCKETS, morton=True, profiling=True)
        rows = []
        for count in (1024, 4096, 6144, 8192):
            frames = [sphere(count, 0.5, (0.0, 0.0, 0.0), seed=k) + torch.tensor([0.03 * k, 0, 0], device="cuda") for k in range(12)]
            runner.reset()
            times = []
            for k, frame in enumerate(frames):
                runner.begin_profile_window()
                runner.stage_world(frame)
                out = runner.replay_next()
                torch.cuda.synchronize()
                if out is not None and k >= 3:
                    prof = runner.resolve_profile_window()
                    times.append(prof.get("encode_ms", 0.0) + prof.get("decode_ms", 0.0))
            rows.append(f"{count}: {np.median(times):.2f} ms")
        print(f"[timing] {name}: encode+decode per pair  " + " | ".join(rows))
        del runner
        torch.cuda.empty_cache()


@torch.inference_mode()
def human(model, root: str) -> None:
    """Whole runner on the recorded human pairs (dense 1 cm voxel points -> buckets), EPE vs pseudo GT."""
    bins = [0, 0.002, 0.01, 0.02, 0.05, 1.0]
    runner = make_runner(model, 8192, BUCKETS, morton=True, auto_scale=True, profiling=True)
    err, gt_norm, used, times = [], [], [], []
    for scene in ("S1_static", "S2_human", "S3_two_sides"):
        files = sorted(glob.glob(f"{root}/{scene}/gap1/pair_*.npz"))
        for f in files:
            z, g = np.load(f), np.load(f.replace("pair_", "gt_"))
            p1 = torch.from_numpy(z["p1"]).cuda()
            p2 = torch.from_numpy(z["p2"]).cuda()
            runner.reset()
            runner.stage_world(p1)
            runner.replay_next()
            runner.begin_profile_window()
            runner.stage_world(p2)
            assert runner.replay_next() is not None
            torch.cuda.synchronize()
            prof = runner.resolve_profile_window()
            times.append(prof.get("encode_ms", 0.0) + prof.get("decode_ms", 0.0))
            flow = runner.flow_world()[0].cpu().numpy()
            sel = runner.source_selection_indices().cpu().numpy()
            keep = g["matched"][sel]
            err.append(np.linalg.norm(flow - g["disp"][sel], axis=1)[keep])
            gt_norm.append(np.linalg.norm(g["disp"][sel], axis=1)[keep])
            used.append(runner.current_point_count())
    e, n = np.concatenate(err), np.concatenate(gt_norm)
    b = np.digitize(n, bins) - 1
    per_bin = " | ".join(f"[{bins[i]*1e3:.0f}-{bins[i+1]*1e3:.0f}] {e[b == i].mean()*1e3:.2f}" for i in range(len(bins) - 1) if (b == i).any())
    counts = {int(c): used.count(c) for c in sorted(set(used))}
    print(f"[human] EPE {e.mean()*1e3:.2f} mm by true motion (mm): {per_bin} | buckets used {counts} | encode(1)+decode median {np.median(times):.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default=None, help="directory with <scene>/gap1/pair_*.npz + gt_*.npz")
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    test_preprocessor()
    plain = build_model()
    test_runner_equivalence(plain)
    test_runner_streaming(plain)
    dense = build_model(fast_top=4096, hier=4096)
    if not args.skip_timing:
        timing(plain, dense)
    if args.human:
        human(dense, args.human)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
