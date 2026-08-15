#!/usr/bin/env python3
"""Correctness/speed smoke test for deployment CUDA recovery kernels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from difflow3d.ops.pointnet2 import pointnet2_utils
from difflow3d.runtime import SoftmaxAnchorMotionRecoverer


def _time_ms(fn, warmup: int = 10, reps: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / reps


def _flow_error(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    error = torch.linalg.vector_norm(a - b, dim=1)
    return (
        float(error.mean().item()),
        float(torch.quantile(error, 0.95).item()),
        float(error.max().item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--queries", type=int, default=96000)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--radius-sigma", type=float, default=4.0)
    parser.add_argument("--hash-size-factor", type=float, default=4.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA.")

    required_symbols = (
        "gaussian_softmax_recovery_wrapper",
        "gaussian_recovery_hash_build_wrapper",
        "gaussian_softmax_recovery_local_wrapper",
    )
    missing = [
        name
        for name in required_symbols
        if not hasattr(pointnet2_utils.pointnet2, name)
    ]
    if missing:
        raise RuntimeError(
            "pointnet2_cuda is missing DifFlow3D recovery kernels. "
            f"Loaded: {pointnet2_utils.extension_path()}. Missing: {missing}. "
            "Run: bash scripts/build_pointnet2_ops.sh"
        )
    print(f"pointnet2_cuda: {pointnet2_utils.extension_path()}")

    torch.manual_seed(7)
    queries = (torch.rand(args.queries, 3, device=device) - 0.5) * 1.0
    anchors = (torch.rand(args.points, 3, device=device) - 0.5) * 1.0
    anchor_flow = torch.randn(args.points, 3, device=device) * 0.01
    dt_s = 1.0 / 30.0

    recoverers = {
        name: SoftmaxAnchorMotionRecoverer(
            chunk_size=4096,
            softmax_sigma_m=args.sigma,
            backend=name,
            local_radius_sigma=args.radius_sigma,
            local_hash_size_factor=args.hash_size_factor,
        )
        for name in ("torch", "global", "local")
    }
    results = {
        name: recoverer.recover(
            query_points=queries,
            anchor_points=anchors,
            anchor_flow=anchor_flow,
            dt_s=dt_s,
        )
        for name, recoverer in recoverers.items()
    }
    torch.cuda.synchronize()

    global_max_error = float(
        (results["global"].flow - results["torch"].flow).abs().max().item()
    )
    local_mean, local_p95, local_max = _flow_error(
        results["local"].flow, results["torch"].flow
    )

    counts = results["local"].local_neighbor_counts
    assert counts is not None
    counts_f = counts.float()
    count_stats = {
        "mean": float(counts_f.mean().item()),
        "median": float(counts_f.median().item()),
        "p95": float(torch.quantile(counts_f, 0.95).item()),
        "max": int(counts.max().item()),
        "empty_ratio": float((counts == 0).float().mean().item()),
    }

    timings = {
        name: _time_ms(
            lambda r=recoverer: r.recover(
                query_points=queries,
                anchor_points=anchors,
                anchor_flow=anchor_flow,
                dt_s=dt_s,
            ),
            reps=20,
        )
        for name, recoverer in recoverers.items()
    }

    print("Gaussian-softmax recovery")
    print(f"  global vs torch max abs:    {global_max_error:.8e} m")
    print(
        "  local vs exact flow EPE:   "
        f"mean={local_mean:.8e} p95={local_p95:.8e} max={local_max:.8e} m"
    )
    print(
        "  local neighbors:           "
        f"mean={count_stats['mean']:.2f} "
        f"median={count_stats['median']:.1f} "
        f"p95={count_stats['p95']:.1f} "
        f"max={count_stats['max']} "
        f"empty={count_stats['empty_ratio']:.6f}"
    )
    for name in ("global", "local", "torch"):
        print(f"  {name:8s}:                   {timings[name]:.3f} ms")

    if global_max_error > 5.0e-5:
        raise RuntimeError("Global Gaussian recovery validation failed.")


if __name__ == "__main__":
    main()
