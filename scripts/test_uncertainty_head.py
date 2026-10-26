#!/usr/bin/env python3
"""Flow-uncertainty head: port vs reference, checkpoint, captured vs eager features, feature definition vs the upstream
model, one stream through the runner (no reset), runner end-to-end and decode latency.

Inside the perceptive-safety-filter container (umbrella checkout at /workspace, weights at /workspace/checkpoints):
    PYTHONPATH=/workspace/external/ScenePredictor/DifFlow3D /opt/tracking-venv/bin/python scripts/test_uncertainty_head.py \\
        --valset real2_stab30_human_rep.npz --ref-tools /ref/hm_tools --ref-dfu /ref/dfu

--valset: canonical real pairs (object arrays p1, p2, gt in model space, k = model units per metre per pair, meta);
          synthetic clouds are used without it (no feature-definition or coverage checks).
--ref-tools / --ref-dfu: read-only checkouts of the human-motion tools (hm/canon_head.py, reference head) and of
          DifFlow3D-uncertainty (upstream model_difflow.py, cov/); the reference checks are skipped without them.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from difflow3d.model.difflow import PointConvBidirection  # noqa: E402
from difflow3d.model.pointconv import knn_point  # noqa: E402
from difflow3d.runtime.checkpoint import load_checkpoint  # noqa: E402
from difflow3d.runtime.normalization import FT3D_MODEL_CENTER, AnchoredNormalization  # noqa: E402
from difflow3d.runtime.runner import DifFlow3DStreamingCudaGraphRunner, configure_fast_inference  # noqa: E402
from difflow3d.runtime.uncertainty import load_flow_covariance_head  # noqa: E402

CHECKPOINTS = Path("/workspace/checkpoints/difflow3d_zu")
BUCKETS = (1024, 2048, 3072, 4096)
ITERS = (4, 2, 2)              # coarse, middle, fine: the schedule the head was trained behind
DENSE_MIN_POINTS = 4096        # configs/difflow_zu.yaml fast_top_level_min_points / hier_cosine_min_points
SEED = 1234
C_REF = np.asarray(FT3D_MODEL_CENTER, np.float32)
RESULTS: dict[str, object] = {}


def eq_radius(chol: torch.Tensor) -> torch.Tensor:
    """det(L L^T)^(1/6): radius of the ball with the ellipsoid's volume (same units as L)."""
    return chol.diagonal(dim1=-2, dim2=-1).prod(dim=-1).abs().pow(1.0 / 3.0)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten() - a.double().mean()
    b = b.double().flatten() - b.double().mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def quantiles(x: torch.Tensor, qs=(0.5, 0.95, 1.0)) -> list[float]:
    x = x.double().flatten()
    return [float(torch.quantile(x, q)) for q in qs]


def fmt(values, digits=2) -> str:
    return "/".join(f"{v:.{digits}f}" for v in values)


# ---------------------------------------------------------------------------
# Models and data
# ---------------------------------------------------------------------------

def build_model(backbone: Path, level1_points: int = 1024):
    """As scene_pred_pipeline/flow_prediction.py builds it (iterations 4/2/2, dense shortcuts, BN running stats off)."""
    coarse, middle, fine = ITERS
    model = PointConvBidirection(iters=max(ITERS), coarse_iters=coarse, middle_iters=middle, fine_iters=fine,
                                 level1_points=level1_points)
    report = load_checkpoint(model, backbone, strict=False)
    model = model.cuda().eval()
    model.configure_dense_points(fast_top_level_min_points=DENSE_MIN_POINTS, hier_cosine_min_points=DENSE_MIN_POINTS)
    for layer in model.modules():
        if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            layer.track_running_stats = False
    return model, report


def load_upstream(ref_dfu: Path, backbone: Path):
    """Upstream DifFlow3D (model_difflow.PointConvBidirection) with the reference feature tap (cov.backbone)."""
    from difflow3d.ops.pointnet2 import pointnet2_cuda

    # The upstream pointnet2_utils imports a top-level pointnet2_cuda; the fork's build has the same wrappers.
    sys.modules.setdefault("pointnet2_cuda", pointnet2_cuda)
    if importlib.util.find_spec("thop") is None:
        # model_difflow imports the FLOP counter at module level (used only in its __main__ block).
        thop = types.ModuleType("thop")
        thop.profile = thop.clever_format = None
        sys.modules["thop"] = thop
    if str(ref_dfu) not in sys.path:
        sys.path.append(str(ref_dfu))
    import model_difflow
    from cov.backbone import FeatureTap, run_backbone

    model = model_difflow.PointConvBidirection(iters=4)
    model.load_state_dict(torch.load(backbone, map_location="cpu"))
    model.recurrent2.iters, model.recurrent1.iters, model.recurrent0.iters = ITERS
    model.cuda()
    for layer in model.modules():
        if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            layer.track_running_stats = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, FeatureTap(model), run_backbone


def load_reference_head(ref_tools: Path, ref_dfu: Path, path: Path):
    for root in (ref_dfu, ref_tools):
        if str(root) not in sys.path:
            sys.path.append(str(root))
    from cov.head import knn_idx
    from hm.canon_head import load_canon_head

    head, _ = load_canon_head(str(path))
    return head, knn_idx


class Pairs:
    """Canonical real pairs (model space, k = model units per metre)."""

    def __init__(self, path: Path) -> None:
        z = np.load(path, allow_pickle=True)
        self.p1, self.p2, self.gt = z["p1"], z["p2"], z["gt"]          # read each member once
        self.k = np.asarray(z["k"], np.float64)
        self.meta = json.loads(str(z["meta"]))["pairs"]
        self.takes = sorted({m["take"] for m in self.meta})

    def of_take(self, take: str, *, min_points: int = 0, max_points: int | None = None) -> list[int]:
        return [i for i, m in enumerate(self.meta)
                if m["take"] == take and min(len(self.p1[i]), len(self.p2[i])) >= min_points
                and (max_points is None or len(self.p1[i]) <= max_points)]

    def get(self, i: int, count: int | None = None, seed: int = 0):
        """(p1, p2, gt) float32 model space, subsampled to ``count`` points (p1/gt and p2 independently)."""
        p1, p2, gt = (np.asarray(a, np.float32) for a in (self.p1[i], self.p2[i], self.gt[i]))
        if count is not None and count < len(p1):
            rng = np.random.default_rng(seed)
            keep1 = np.sort(rng.choice(len(p1), count, replace=False))
            keep2 = np.sort(rng.choice(len(p2), count, replace=False))
            p1, gt, p2 = p1[keep1], gt[keep1], p2[keep2]
        return p1, p2, gt


def to_world(p_model: np.ndarray, k: float) -> torch.Tensor:
    return torch.from_numpy((p_model - C_REF) / np.float32(k)).cuda()


def synthetic_world_pair(count: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    body = torch.randn(count, 3, generator=g) * torch.tensor([0.15, 0.4, 0.1]) + torch.tensor([0.0, 0.0, 1.5])
    return body.cuda(), (body + torch.tensor([0.01, 0.0, 0.0])).cuda()


def fork_features(model, p1: torch.Tensor, p2: torch.Tensor, seed: int):
    """Eager fork decode (fast eval path, as captured) -> flow [B,N,3], feat [B,N,128], u [B,N,1] (channels-last)."""
    unit = model.recurrent0
    with torch.inference_mode():
        source = model.encode_frame(p1, p1)
        target = model.encode_frame(p2, p2)
        unit.capture_head_inputs = True
        try:
            torch.cuda.manual_seed(seed)
            output = model.decode_pair(source, target, None, 0.1)
            feat, certainty = unit.head_inputs
        finally:
            unit.capture_head_inputs = False
            unit.head_inputs = None
        return output[0][0][0].transpose(1, 2), feat.transpose(1, 2), certainty.transpose(1, 2)


class RecordingHead(torch.nn.Module):
    """Pass-through wrapper that keeps the static input tensors of each captured head call (test only). ``kk`` is
    keyword-only, as in the runner's head call contract."""

    def __init__(self, head) -> None:
        super().__init__()
        self.head = head
        self.captured: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    def forward(self, xyz, feat, u, flow, *, kk):
        if torch.cuda.is_current_stream_capturing():
            count = int(xyz.shape[1])
            direction = "ba" if (count, "ab") in self.captured else "ab"
            self.captured[(count, direction)] = {"xyz": xyz, "feat": feat, "u": u, "flow": flow, "kk": kk}
        return self.head(xyz, feat, u, flow, kk=kk)


def make_runner(model, head):
    return DifFlow3DStreamingCudaGraphRunner(
        model, batch_size=1, num_points=max(BUCKETS), uncertainty=0.1, warmup=3, dt_s=1.0,
        second_base_voxel_size_m=0.01, second_candidate_ratio=1.1, auto_spatial_scale=False, fixed_spatial_scale=1.0,
        final_selection="uniform", point_buckets=BUCKETS, sort_anchors_morton=True,
        normalization=AnchoredNormalization(), uncertainty_head=head)


def allocations() -> int:
    return int(torch.cuda.memory_stats()["allocation.all.allocated"])


def stream_pair(runner, w1: torch.Tensor, w2: torch.Tensor, seed: int) -> int:
    """reset, stage + replay both frames (seeded decode); returns caching-allocator allocations during replay_next."""
    runner.reset()
    runner.stage_world(w1)
    runner.replay_next()
    runner.stage_world(w2)
    torch.cuda.manual_seed(seed)
    before = allocations()
    runner.replay_next()
    return allocations() - before


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_strict_load(report) -> None:
    print(f"[b] ZU backbone -> fork PointConvBidirection(4/2/2): missing={list(report.missing_keys)} "
          f"unexpected={list(report.unexpected_keys)}")
    assert report.is_exact, "the ZU checkpoint does not load strictly"
    RESULTS["b"] = {"missing": list(report.missing_keys), "unexpected": list(report.unexpected_keys)}
    print("[ok] b: strict load")


def compare_heads(name, port, reference, knn_idx, xyz, feat, u, flow, kk) -> dict:
    with torch.no_grad():
        idx = knn_idx(xyz, reference.k)
        l_ref = reference(xyz, feat, u, flow, idx=idx, kk=kk)
        l_same = port(xyz, feat, u, flow, kk, idx=idx)
        l_own = port(xyz, feat, u, flow, kk)
        own_idx = knn_point(port.k, xyz.contiguous(), xyz.contiguous())
    scale = float(l_ref.abs().max())
    same_set = (own_idx.sort(-1).values == idx.sort(-1).values).all(-1)
    r_ref, r_own = eq_radius(l_ref), eq_radius(l_own)
    rel_r = ((r_own - r_ref).abs() / r_ref)
    row = {
        "max_abs_same_idx": float((l_same - l_ref).abs().max()),
        "max_rel_same_idx": float((l_same - l_ref).abs().max()) / scale,
        "knn_identical_rows": float(same_set.float().mean()),
        "max_abs_own_knn_identical_rows": float((l_own - l_ref).abs()[same_set].max()) if same_set.any() else 0.0,
        "max_abs_own_knn": float((l_own - l_ref).abs().max()),
        "r_rel_own_knn_max": float(rel_r.max()),
        "L_abs_max": scale,
    }
    print(f"    {name}: same kNN max|dL|={row['max_abs_same_idx']:.3g} (rel {row['max_rel_same_idx']:.2g}); "
          f"own fused kNN: identical sets {100 * row['knn_identical_rows']:.3f}% rows, max|dL| {row['max_abs_own_knn']:.3g} "
          f"(identical rows {row['max_abs_own_knn_identical_rows']:.3g}), max rel dr {row['r_rel_own_knn_max']:.2g}")
    return row


def check_port(port, reference, knn_idx, model, pairs: Pairs | None) -> None:
    print("[a] port vs reference NormCovHead (fp32, TF32 off)")
    rows = {}
    g = torch.Generator(device="cuda").manual_seed(0)
    for count in (1024, 4096):
        # Random inputs, batch 2 with different scales (canonical-like cloud around C_REF, RMS radius 8.97).
        xyz = torch.randn(2, count, 3, device="cuda", generator=g)
        xyz = xyz / xyz.square().sum(-1).mean().sqrt() * 8.97 + torch.from_numpy(C_REF).cuda()
        feat = torch.randn(2, count, 128, device="cuda", generator=g)
        u = torch.randn(2, count, 1, device="cuda", generator=g)
        kk = torch.tensor([12.0, 33.0], device="cuda")
        flow = torch.randn(2, count, 3, device="cuda", generator=g) * torch.rand(2, count, 1, device="cuda", generator=g) \
            * 0.06 * kk.view(-1, 1, 1)
        rows[f"random_{count}"] = compare_heads(f"random   N={count} B=2", port, reference, knn_idx, xyz, feat, u, flow, kk)
        if pairs is not None:
            i = pairs.of_take(pairs.takes[-1], min_points=4096)[100]
            p1, p2, _ = pairs.get(i, count)
            p1t, p2t = torch.from_numpy(p1).cuda()[None], torch.from_numpy(p2).cuda()[None]
            flow_f, feat_f, u_f = fork_features(model, p1t, p2t, SEED)
            kk = torch.tensor([pairs.k[i]], device="cuda", dtype=torch.float32)
            rows[f"real_{count}"] = compare_heads(f"real     N={count} pair {i}", port, reference, knn_idx,
                                                  p1t, feat_f, u_f, flow_f, kk)
            if count == 4096:
                configure_fast_inference(True)
                with torch.no_grad():
                    l_tf32 = port(p1t, feat_f, u_f, flow_f, kk, idx=knn_idx(p1t, 16))
                configure_fast_inference(False)
                with torch.no_grad():
                    l_ref = reference(p1t, feat_f, u_f, flow_f, idx=knn_idx(p1t, 16), kk=kk)
                rel = (eq_radius(l_tf32) - eq_radius(l_ref)).abs() / eq_radius(l_ref)
                rows["tf32_r_rel"] = quantiles(rel)
                print("    TF32 port vs fp32 reference, eq-radius rel diff p50/p95/max = "
                      + "/".join(f"{v:.2g}" for v in rows["tf32_r_rel"]))
    for name, row in rows.items():
        if name.startswith(("random", "real")):
            assert row["max_rel_same_idx"] < 1e-5, (name, row)
            assert row["knn_identical_rows"] > 0.95, (name, row)
    RESULTS["a"] = rows
    print("[ok] a: port == reference")


@contextmanager
def aligned_noise(seed: int):
    """``torch.randn_like`` drawn in logical (contiguous) order from one seeded generator, so the fork and the upstream
    model, whose tensors have different memory layouts, get identical diffusion noise (test only)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    original = torch.randn_like

    def randn_like(x, *args, **kwargs):
        return torch.randn(tuple(x.shape), device=x.device, generator=generator).to(x.dtype)

    torch.randn_like = randn_like
    try:
        yield
    finally:
        torch.randn_like = original


def feature_stats(flow, feat, u, chol, ref, k: float, gt=None) -> dict:
    """Fork pipeline (flow, feat, u, chol) vs a reference pipeline ``ref`` (same tuple) on one pair."""
    mm = 1000.0 / k
    r, r_ref = eq_radius(chol) * mm, eq_radius(ref[3]) * mm
    row = {
        "flow_diff_mm": quantiles((flow - ref[0]).norm(dim=-1) * mm),
        "feat_corr": [pearson(feat[..., a:b], ref[1][..., a:b]) for a, b in ((0, 64), (64, 96), (96, 128))],
        "u_corr": pearson(u, ref[2]),
        "r_logcorr": pearson(r.log(), r_ref.log()),
        "r_ratio": quantiles(r / r_ref, (0.05, 0.5, 0.95)),
        "r_mm": quantiles(r, (0.05, 0.5, 0.95)),
    }
    if gt is not None:
        valid = torch.isfinite(gt).all(-1)
        row["epe_mm"] = float(((flow - gt).norm(dim=-1) * mm)[valid].mean())
    return row


def check_features_vs_upstream(model, port, reference, upstream, pairs: Pairs, backbone: Path) -> None:
    """Fork fast eval path + capture + port head vs upstream model + FeatureTap + reference head, identical noise.

    The upstream encoder samples 2048 level-1 points, the fork 1024 by default (``level1_points``). The capture is
    asserted on a fork model with 2048 (the configuration equal to upstream); the runner model (deployed: 1024, a
    different backbone setting whose calibration tables are fitted through the deployed path) is only reported.
    """
    up_model, tap, run_backbone = upstream
    print("[d] feature definition vs upstream (FeatureTap + reference head): fp32, dense shortcuts off, identical "
          "diffusion noise; seed floor = upstream with other noise")
    matched, _ = build_model(backbone, level1_points=2048)
    for m in (matched, model):
        m.configure_dense_points(fast_top_level_min_points=0, hier_cosine_min_points=0)
    rows = []
    try:
        for take in pairs.takes:
            ids = pairs.of_take(take, min_points=4096)
            i = ids[len(ids) // 2]
            p1, p2, gt = pairs.get(i, 4096)
            k = float(pairs.k[i])
            p1t, p2t = torch.from_numpy(p1).cuda()[None], torch.from_numpy(p2).cuda()[None]
            gt_t = torch.from_numpy(gt).cuda()[None]
            kk = torch.tensor([k], device="cuda", dtype=torch.float32)

            def fork(m, seed):
                with aligned_noise(seed):
                    flow, feat, u = fork_features(m, p1t, p2t, seed)
                with torch.no_grad():
                    return flow, feat, u, port(p1t, feat, u, flow, kk)

            def upstream_run(seed):
                with aligned_noise(seed), torch.no_grad():
                    out = run_backbone(up_model, tap, "s", p1t, p2t, p1t, p2t)
                    return out.flow, out.feat, out.u, reference(p1t, out.feat, out.u, out.flow, kk=kk)

            up, up_other = upstream_run(SEED), upstream_run(SEED + 1)
            row = {
                "pair": i, "take": take, "k": k,
                "matched": feature_stats(*fork(matched, SEED), up, k, gt_t),
                "deployed": feature_stats(*fork(model, SEED), up, k, gt_t),
                "seed_floor": feature_stats(*up_other, up, k),
            }
            row["upstream_epe_mm"] = feature_stats(*up, up, k, gt_t)["epe_mm"]
            rows.append(row)
            print(f"    {take:11s} pair {i:4d} (upstream EPE {row['upstream_epe_mm']:.2f} mm)")
            for name in ("matched", "deployed", "seed_floor"):
                s = row[name]
                print(f"      {name:10s} |dflow| mm p50/p95/max {fmt(s['flow_diff_mm'])}; feat corr latent/enc/fe "
                      f"{fmt(s['feat_corr'], 4)}; u corr {s['u_corr']:.4f}; r log-corr {s['r_logcorr']:.4f}; "
                      f"r ratio p5/p50/p95 {fmt(s['r_ratio'], 3)}; r mm p5/p50/p95 {fmt(s['r_mm'])}"
                      + (f"; EPE {s['epe_mm']:.2f} mm" if "epe_mm" in s else ""))
    finally:
        model.configure_dense_points(fast_top_level_min_points=DENSE_MIN_POINTS, hier_cosine_min_points=DENSE_MIN_POINTS)
        del matched
    for row in rows:
        s = row["matched"]
        assert min(s["feat_corr"]) > 0.995 and s["u_corr"] > 0.99 and s["r_logcorr"] > 0.99, row
        assert 0.99 < s["r_ratio"][1] < 1.01, row
    summary = {name: float(np.median([row[name]["r_ratio"][1] for row in rows])) for name in ("matched", "deployed")}
    print(f"    median r ratio vs upstream: matched (level1_points=2048, asserted) {summary['matched']:.3f}; runner model "
          f"(level1_points={model.level1_points}, reported only) {summary['deployed']:.3f}")
    RESULTS["d"] = {"pairs": rows, "median_r_ratio": summary}
    print("[ok] d: captured head inputs == upstream FeatureTap definition (level1_points=2048)")


def check_captured_vs_eager(runner, recorder, model, head, clouds) -> None:
    print("[c] captured graph vs eager decode_pair + head (same seed, TF32 as deployed)")
    rows = {}
    for count, (w1, w2) in clouds.items():
        stream_pair(runner, w1, w2, SEED)
        torch.cuda.synchronize()
        bucket = runner._current_bucket
        assert bucket.num_points == count, (bucket.num_points, count)
        direction = "ab" if runner._last_source_slot == 0 else "ba"
        static = recorder.captured[(count, direction)]
        scale = runner._slot_scale(runner._last_source_slot)
        assert runner._current_scale == scale, (runner._current_scale, scale)
        assert abs(float(static["kk"][0]) - scale) <= 1e-6 * scale, (float(static["kk"][0]), scale)
        chol_graph = runner.flow_chol_model().clone()
        flow_graph = runner.flow().clone()
        feat_graph, u_graph = static["feat"].clone(), static["u"].clone()
        source, target = runner.source_points().clone(), runner.target_points().clone()
        with torch.inference_mode():
            chol_static = head(static["xyz"], static["feat"], static["u"], static["flow"], kk=static["kk"])
        flow_e, feat_e, u_e = fork_features(model, source, target, SEED)
        kk = torch.full((1,), scale, device="cuda")
        with torch.inference_mode():
            chol_e = head(source, feat_e, u_e, flow_e, kk=kk)
        row = {
            "head_on_static_vs_graph": float((chol_static - chol_graph).abs().max()),
            "flow": float((flow_e - flow_graph).abs().max()),
            "feat": float((feat_e - feat_graph).abs().max()),
            "u": float((u_e - u_graph).abs().max()),
            "chol": float((chol_e - chol_graph).abs().max()),
            "chol_abs_max": float(chol_graph.abs().max()),
            "kk": scale,
        }
        rows[count] = row
        print(f"    bucket {count}: head(static inputs) vs graph L max|d| {row['head_on_static_vs_graph']:.3g}; eager vs graph "
              f"max|d| flow {row['flow']:.3g}, feat {row['feat']:.3g}, u {row['u']:.3g}, L {row['chol']:.3g} "
              f"(max|L| {row['chol_abs_max']:.3g}, kk {scale:.3f})")
        assert row["head_on_static_vs_graph"] <= 1e-6 * row["chol_abs_max"], row
        assert row["chol"] <= 1e-4 * row["chol_abs_max"], row
    RESULTS["c"] = rows
    print("[ok] c: captured head inputs and L == eager")


# (anchor count, scale about the centroid) per frame. Counts stay below the voxel-2 threshold (1.1 x 4096), so the
# natural buckets are 4096 / 2048 / 3072; a 2.2x radius change re-anchors (scale_ratio 1.5).
STREAM_PLAN = ((4400, 1.0), (4400, 1.0), (2150, 1.0), (4400, 2.2), (4400, 2.2), (3150, 1.0), (4400, 1.0), (4400, 1.0))


def stream_frames(pairs: Pairs | None) -> list[torch.Tensor]:
    """World frames of one stream: alternately the source and target cloud of one pair, subsampled and scaled per
    STREAM_PLAN."""
    frames = []
    if pairs is not None:
        ids = pairs.of_take(pairs.takes[-1], min_points=max(c for c, _ in STREAM_PLAN))
        i = ids[len(ids) // 2]
        k = float(pairs.k[i])
    for j, (count, scale) in enumerate(STREAM_PLAN):
        if pairs is not None:
            p1, p2, _ = pairs.get(i, count, seed=j)
            frame = to_world(p1 if j % 2 == 0 else p2, k)
        else:
            frame = synthetic_world_pair(count, j)[j % 2]
        center = frame.mean(0)
        frames.append((frame - center) * scale + center)
    return frames


def check_stream(runner, recorder, model, head, pairs: Pairs | None) -> None:
    """One stream without reset: per decode, kk == source scale, graph L == head on the captured inputs and on an
    eager decode, flow_cov_world == L L^T / k^2 with the decode-time k; the same values are read again after the next
    frame is staged (which may re-anchor and re-stage the source slot) and before it is decoded."""
    print("[g] one stream without reset (bucket changes, re-anchors, ab/ba alternating), read before and after staging "
          "the next frame")
    runner.reset()
    reanchors, reencodes = runner.normalization.reanchor_count, runner.reencode_count
    rows, pending = [], None
    for j, frame in enumerate(stream_frames(pairs)):
        runner.stage_world(frame)
        if pending is not None:
            # Staged, not decoded: the source slot may hold the new frame and transform now.
            row, expected = pending
            row["slot_scale_after_stage"] = runner._slot_scale(row["source_slot"])
            row["stage_changed_scale"] = row["slot_scale_after_stage"] != row["k"]
            row["after_stage_equal"] = (torch.equal(runner.flow_world(), expected["flow_world"])
                                        and torch.equal(runner.flow_cov_world(), expected["cov"]))
            assert row["after_stage_equal"], row
            pending = None
        torch.cuda.manual_seed(SEED + j)
        if runner.replay_next() is None:
            continue
        torch.cuda.synchronize()
        source_slot = runner._last_source_slot
        direction = "ab" if source_slot == 0 else "ba"
        count = runner.current_point_count()
        k = runner._slot_scale(source_slot)
        kk = float(runner._head_scale[0])
        chol = runner.flow_chol_model().clone()
        flow_graph = runner.flow().clone()
        static = recorder.captured[(count, direction)]
        with torch.inference_mode():
            chol_static = head(static["xyz"], static["feat"], static["u"], static["flow"], kk=static["kk"])
        source, target = runner.source_points().clone(), runner.target_points().clone()
        flow_e, feat_e, u_e = fork_features(model, source, target, SEED + j)
        with torch.inference_mode():
            chol_e = head(source, feat_e, u_e, flow_e, kk=torch.full((1,), k, device="cuda"))
        cov = runner.flow_cov_world()
        cov_ref = (chol.double() @ chol.double().transpose(-1, -2)) / (k * k)
        flow_world = runner.flow_world()
        chol_max, flow_max = float(chol.abs().max()), float(flow_graph.abs().max())
        row = {
            "frame": j, "bucket": count, "direction": direction, "source_slot": source_slot, "k": k, "kk": kk,
            "reanchors": runner.normalization.reanchor_count - reanchors, "reencodes": runner.reencode_count - reencodes,
            "head_on_static_vs_graph": float((chol_static - chol).abs().max()),
            "eager_vs_graph_L": float((chol_e - chol).abs().max()), "eager_vs_graph_flow": float((flow_e - flow_graph).abs().max()),
            "chol_abs_max": chol_max, "cov_rel": float((cov.double() - cov_ref).abs().max() / cov_ref.abs().max()),
        }
        rows.append(row)
        print(f"    frame {j}: bucket {count} {direction}, re-anchors {row['reanchors']}, re-encodes {row['reencodes']}, "
              f"k {k:.4f}, |kk-k|/k {abs(kk - k) / k:.1e}; L: head(static) vs graph {row['head_on_static_vs_graph']:.2g}, "
              f"eager vs graph {row['eager_vs_graph_L']:.2g} (max|L| {chol_max:.3g}); eager vs graph flow "
              f"{row['eager_vs_graph_flow']:.2g}; cov vs L L^T/k^2 rel {row['cov_rel']:.2g}")
        assert runner._current_scale == k and abs(kk - k) <= 1e-6 * k, row
        assert row["head_on_static_vs_graph"] <= 1e-6 * chol_max, row
        assert row["eager_vs_graph_L"] <= 1e-4 * chol_max and row["eager_vs_graph_flow"] <= 1e-4 * flow_max, row
        assert row["cov_rel"] <= 2e-6 and torch.equal(cov, cov.transpose(-1, -2)), row
        assert torch.equal(flow_world, flow_graph / k), "flow_world != flow / decode-time scale"
        pending = (row, {"flow_world": flow_world, "cov": cov})
    for previous, row in zip(rows, rows[1:]):
        assert previous["direction"] != row["direction"], "decode directions did not alternate"
    changed = [row["frame"] for row in rows if row.get("stage_changed_scale")]
    buckets = [row["bucket"] for row in rows]
    print(f"    buckets {buckets}; staging the next frame changed the source slot's scale after the decode of frames "
          f"{changed} (the reads above use the decode-time scale)")
    assert rows[-1]["reanchors"] == 2 and rows[-1]["reencodes"] > 0, rows[-1]
    assert len(set(buckets)) >= 3 and changed, (buckets, changed)
    RESULTS["g"] = rows
    print("[ok] g: stream: kk, L and the world covariance follow the decode-time scale")


def check_runner(runner, plain, pairs: Pairs | None, calibration: dict | None) -> None:
    print("[e] runner end-to-end (AnchoredNormalization, buckets %s)" % (BUCKETS,))
    assert runner.has_uncertainty and not plain.has_uncertainty
    runner.reset()
    for method in (runner.flow_cov_world, runner.flow_chol_model, plain.flow_cov_world, plain.flow_chol_model):
        try:
            method()
            raise AssertionError(f"{method.__name__} did not raise")
        except RuntimeError:
            pass

    streams = []
    if pairs is not None:
        for take in pairs.takes:
            ids = pairs.of_take(take, max_points=4096)
            if not ids:
                continue
            for j in np.linspace(0, len(ids) - 1, 8).astype(int):
                streams.append((ids[j], None))
            for j in np.linspace(0, len(ids) - 1, 2).astype(int):
                streams.append((ids[j], 1500))
    else:
        streams = [(seed, count) for seed, count in enumerate((1500, 2100, 3100, 4096))]

    alloc_head, alloc_plain, radii, residuals, buckets = 0, 0, [], [], {}
    for n, (i, count) in enumerate(streams):
        if pairs is not None:
            p1, p2, gt = pairs.get(i, count, seed=n)
            k = float(pairs.k[i])
            w1, w2 = to_world(p1, k), to_world(p2, k)
            gt_world = torch.from_numpy(gt / np.float32(k)).cuda()
        else:
            w1, w2 = synthetic_world_pair(count, n)
            gt_world = None
        alloc_head += stream_pair(runner, w1, w2, SEED + n)
        alloc_plain += stream_pair(plain, w1, w2, SEED + n)
        flow_head, flow_plain = runner.flow_world(), plain.flow_world()
        assert torch.equal(flow_head, flow_plain), "the head changed the flow"
        cov = runner.flow_cov_world()
        assert cov.shape == (*flow_head.shape[:2], 3, 3) and cov.dtype == torch.float32
        assert cov.shape[:2] == runner.source_points_world().shape[:2]
        assert torch.isfinite(cov).all()
        assert torch.equal(cov, cov.transpose(-1, -2)), "covariance not symmetric"
        assert int(torch.linalg.cholesky_ex(cov.double())[1].max()) == 0, "covariance not positive definite"
        scale = runner._slot_scale(runner._last_source_slot)
        assert abs(float(runner._head_scale[0]) - scale) <= 1e-6 * scale
        radius = torch.linalg.det(cov.double()).pow(1.0 / 6.0)[0]
        radii.append(radius)
        buckets[runner.current_point_count()] = buckets.get(runner.current_point_count(), 0) + 1
        if gt_world is not None:
            selection = runner.source_selection_indices()
            assert torch.allclose(runner.source_points_world()[0], w1[selection], atol=2e-5)
            d = (gt_world[selection] - flow_head[0]).norm(dim=-1)
            valid = torch.isfinite(d)
            residuals.append((d[valid].double(), radius[valid]))
    radii_mm = torch.cat(radii) * 1000.0
    row = {
        "pairs": len(streams), "buckets": buckets, "replay_allocations_head": alloc_head,
        "replay_allocations_plain": alloc_plain,
        "eq_radius_mm_p1_p50_p99": quantiles(radii_mm, (0.01, 0.5, 0.99)),
        "eq_radius_mm_min_max": [float(radii_mm.min()), float(radii_mm.max())],
    }
    print(f"    {len(streams)} pairs, buckets {buckets}; flow bit-identical with/without head; cov finite, symmetric, SPD; "
          f"eq-radius mm p1/p50/p99 {fmt(row['eq_radius_mm_p1_p50_p99'])}, min/max {fmt(row['eq_radius_mm_min_max'])}; "
          f"allocations in replay_next: head {alloc_head}, plain {alloc_plain}")
    assert alloc_head == 0 and alloc_plain == 0
    assert 0.1 < row["eq_radius_mm_p1_p50_p99"][1] < 50.0
    if residuals and calibration is not None:
        d = torch.cat([x for x, _ in residuals])
        r = torch.cat([y for _, y in residuals])
        coverage = {}
        for key, c in calibration["scale_after"].items():
            coverage[key] = float((d <= c * r).double().mean())
        reference = {key: calibration["after"]["human"][f"rep_{key}"]["all"]["own"] for key in calibration["scale_after"]}
        row["coverage_global"] = coverage
        row["coverage_reference_rep"] = reference
        row["epe_mm"] = float(d.mean() * 1000.0)
        row["gt_points"] = int(d.numel())
        # Reference ball scales of the evaluation preprocessing (--calibration), not the deployed tables (those are
        # fitted through the deployed path); reported only.
        print(f"    coverage of the reference global ball scales (--calibration, evaluation preprocessing; runner, "
              f"{d.numel()} GT points, EPE {row['epe_mm']:.2f} mm): "
              + ", ".join(f"{key}: {coverage[key]:.3f} (ref rep {reference[key]:.3f})" for key in coverage))
    RESULTS["e"] = row
    print("[ok] e: runner end-to-end")


def check_latency(runner, plain, clouds, reps: int = 100) -> None:
    print("[f] decode graph replay latency (CUDA events, %d reps after 10 warm-up)" % reps)
    rows = {}
    for count, (w1, w2) in clouds.items():
        row = {}
        for name, r in (("plain", plain), ("head", runner)):
            stream_pair(r, w1, w2, SEED)
            bucket = r._current_bucket
            graph = bucket.decode_graph_ab if r._last_source_slot == 0 else bucket.decode_graph_ba
            for _ in range(10):
                graph.replay()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            for start, end in events:
                start.record()
                graph.replay()
                end.record()
            torch.cuda.synchronize()
            times = torch.tensor([start.elapsed_time(end) for start, end in events])
            row[name] = [float(times.median()), float(times.mean())]
        row["delta_median_ms"] = row["head"][0] - row["plain"][0]
        rows[count] = row
        print(f"    {count:5d} pts: no head {row['plain'][0]:.3f} ms (mean {row['plain'][1]:.3f}), "
              f"with head {row['head'][0]:.3f} ms (mean {row['head'][1]:.3f}), delta {row['delta_median_ms']:+.3f} ms")
    RESULTS["f"] = rows


def bucket_clouds(pairs: Pairs | None) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """One world pair per bucket whose natural bucket is that bucket."""
    clouds = {}
    for count in BUCKETS:
        if pairs is not None:
            i = pairs.of_take(pairs.takes[-1], min_points=max(BUCKETS) + 50)[200]
            p1, p2, _ = pairs.get(i, count + 50, seed=count)
            k = float(pairs.k[i])
            clouds[count] = (to_world(p1, k), to_world(p2, k))
        else:
            clouds[count] = synthetic_world_pair(count + 50, count)
    return clouds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", type=Path, default=CHECKPOINTS / "zu_p4-Z2u-zup_model_e09.pth")
    parser.add_argument("--head", type=Path, default=CHECKPOINTS / "zu-U3_head.pt")
    parser.add_argument("--calibration", type=Path, default=CHECKPOINTS / "calib_global_stab30.json")
    parser.add_argument("--valset", type=Path, default=None)
    parser.add_argument("--ref-tools", type=Path, default=None)
    parser.add_argument("--ref-dfu", type=Path, default=None)
    parser.add_argument("--level1-points", type=int, default=1024,
                        help="model level-1 size for the runner checks (fork default 1024, upstream 2048)")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON with all numbers")
    args = parser.parse_args()

    configure_fast_inference(False)
    model, report = build_model(args.backbone, args.level1_points)
    check_strict_load(report)

    head, meta = load_flow_covariance_head(args.head, "cuda")
    assert meta["iters"] == ",".join(str(v) for v in ITERS), meta["iters"]
    print(f"    head: backbone {meta['backbone']}, iters {meta['iters']}, disp_form {meta['disp_form']}, seed {meta['seed']}")
    pairs = Pairs(args.valset) if args.valset is not None else None
    calibration = json.loads(args.calibration.read_text()) if args.calibration.is_file() else None

    have_refs = args.ref_tools is not None and args.ref_dfu is not None
    if have_refs:
        reference, knn_idx = load_reference_head(args.ref_tools, args.ref_dfu, args.head)
        check_port(head, reference, knn_idx, model, pairs)
        if pairs is not None:
            upstream = load_upstream(args.ref_dfu, args.backbone)
            check_features_vs_upstream(model, head, reference, upstream, pairs, args.backbone)
            upstream[1].remove()
            del upstream
        else:
            print("[skip] d: needs --valset")
    else:
        print("[skip] a, d: need --ref-tools and --ref-dfu")

    recorder = RecordingHead(head)
    runner = make_runner(model, recorder)
    plain = make_runner(model, None)
    clouds = bucket_clouds(pairs)
    check_captured_vs_eager(runner, recorder, model, head, {count: clouds[count] for count in (1024, 4096)})
    check_stream(runner, recorder, model, head, pairs)
    check_runner(runner, plain, pairs, calibration)
    check_latency(runner, plain, clouds)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(RESULTS, indent=1, default=str))
    print("ALL UNCERTAINTY HEAD TESTS PASSED")


if __name__ == "__main__":
    main()
