#!/usr/bin/env python3
"""Validate and benchmark DifFlow3D anchor-level temporal CUDA operators."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

from difflow3d.ops.pointnet2 import pointnet2_utils
from difflow3d.runtime.motion import CudaAnchorTemporalOps


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate strict-track transport, velocity-only KF, and CUDA latency."
    )
    parser.add_argument("--anchors", type=int, default=2048)
    parser.add_argument("--tracks", type=int, default=4)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--radius-sigma", type=float, default=4.0)
    parser.add_argument("--hash-size-factor", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--recursive-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-4)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarize(values: list[float]) -> TimingStats:
    return TimingStats(
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        p95_ms=percentile(values, 0.95),
        max_ms=max(values),
    )


def benchmark_cuda(fn, warmup: int, iterations: int) -> TimingStats:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return summarize([start.elapsed_time(end) for start, end in zip(starts, ends)])


def strict_local_transport_reference(
    queries: torch.Tensor,
    sources: torch.Tensor,
    values: torch.Tensor,
    query_track_ids: torch.Tensor,
    source_track_ids: torch.Tensor,
    sigma: float,
    radius_sigma: float,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius2 = (sigma * radius_sigma) ** 2
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    output = torch.zeros(
        (queries.shape[0], values.shape[1]), device=queries.device, dtype=torch.float32
    )
    support = torch.zeros((queries.shape[0],), device=queries.device, dtype=torch.int32)

    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        q = queries[start:end]
        dist2 = torch.cdist(q, sources).square()
        same_track = query_track_ids[start:end, None] == source_track_ids[None, :]
        valid = same_track & (dist2 <= radius2)
        support[start:end] = valid.sum(dim=1, dtype=torch.int32)

        logits = -dist2 * inv_two_sigma2
        logits = logits.masked_fill(~valid, float("-inf"))
        has_support = valid.any(dim=1)
        if has_support.any():
            weights = torch.softmax(logits[has_support], dim=1)
            output_chunk = output[start:end]
            output_chunk[has_support] = weights @ values
    return output, support


def kf_reference(
    current_flow: torch.Tensor,
    previous_state6: torch.Tensor,
    support: torch.Tensor,
    dt: float,
    *,
    process_velocity_std_mps: float,
    measurement_noise_std_mps: float,
    initial_velocity_std_mps: float,
    min_innovation_variance: float,
) -> torch.Tensor:
    """Torch reference for the velocity-only diagonal-covariance KF."""
    z = current_flow / dt
    measurement_finite = torch.isfinite(z).all(dim=1)
    has_history = support > 0
    output = torch.empty_like(previous_state6)

    initial_variance = initial_velocity_std_mps**2
    output[:, :3] = torch.where(measurement_finite[:, None], z, torch.zeros_like(z))
    output[:, 3:6] = initial_variance
    if not has_history.any():
        return output

    idx = torch.nonzero(has_history, as_tuple=False).squeeze(1)
    prev = previous_state6[idx]
    local_measurement_finite = measurement_finite[idx]
    history_finite = torch.isfinite(prev).all(dim=1)
    if not history_finite.any():
        return output

    valid_idx = idx[history_finite]
    prev = prev[history_finite]
    local_measurement_finite = local_measurement_finite[history_finite]
    pred_v = prev[:, :3]
    process_variance = process_velocity_std_mps**2
    pred_p = torch.clamp(prev[:, 3:6] + process_variance, min=1.0e-12, max=1.0e12)

    invalid_idx = valid_idx[~local_measurement_finite]
    if invalid_idx.numel() > 0:
        local = ~local_measurement_finite
        output[invalid_idx, :3] = pred_v[local]
        output[invalid_idx, 3:6] = pred_p[local]

    finite_local = local_measurement_finite
    if not finite_local.any():
        return output

    finite_idx = valid_idx[finite_local]
    pred_v_f = pred_v[finite_local]
    pred_p_f = pred_p[finite_local]
    residual = z[finite_idx] - pred_v_f
    measurement_variance = measurement_noise_std_mps**2
    innovation_var = torch.clamp_min(pred_p_f + measurement_variance, min_innovation_variance)
    gain = pred_p_f / innovation_var
    updated_v = pred_v_f + gain * residual
    updated_p = (1.0 - gain).square() * pred_p_f + gain.square() * measurement_variance
    updated_p = torch.clamp(updated_p, min=1.0e-12, max=1.0e12)

    finite_update = torch.isfinite(updated_v).all(dim=1) & torch.isfinite(updated_p).all(dim=1)
    output[finite_idx, :3] = torch.where(finite_update[:, None], updated_v, pred_v_f)
    output[finite_idx, 3:6] = torch.where(finite_update[:, None], updated_p, pred_p_f)
    return output


def make_scene(args: argparse.Namespace, device: torch.device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    count = args.anchors
    track_ids = torch.arange(count, device=device, dtype=torch.int32) % args.tracks

    centers = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.25, 0.25, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    if args.tracks > centers.shape[0]:
        extra = torch.randn(
            (args.tracks - centers.shape[0], 3), generator=generator, device=device
        ) * 0.2
        centers = torch.cat((centers, extra), dim=0)
    centers = centers[: args.tracks]

    sources = centers[track_ids.long()] + 0.02 * torch.randn(
        (count, 3), generator=generator, device=device
    )
    queries = sources + 0.002 * torch.randn((count, 3), generator=generator, device=device)
    query_track_ids = track_ids.clone()
    missing = min(16, count)
    query_track_ids[-missing:] = args.tracks + 100

    values3 = torch.randn((count, 3), generator=generator, device=device)
    values6 = torch.randn((count, 6), generator=generator, device=device)
    return queries, sources, values3, values6, query_track_ids, track_ids


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, args: argparse.Namespace):
    try:
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
    except AssertionError as exc:
        max_error = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} failed; max abs error={max_error:.6g}") from exc
    print(f"[PASS] {name}")


def print_timing(name: str, stats: TimingStats) -> None:
    print(
        f"{name:<24} mean={stats.mean_ms:8.4f} ms  "
        f"median={stats.median_ms:8.4f}  p95={stats.p95_ms:8.4f}  max={stats.max_ms:8.4f}"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not pointnet2_utils.has_gaussian_softmax_transport_local_track_aware():
        raise RuntimeError("Missing temporal transport CUDA symbol; rebuild PointNet2 ops")
    if not pointnet2_utils.has_anchor_velocity_kalman_op():
        raise RuntimeError("Missing velocity-KF CUDA symbol; rebuild PointNet2 ops")

    device = torch.device("cuda")
    print("=" * 96)
    print("DifFlow3D anchor temporal CUDA validation")
    print("=" * 96)
    print(f"GPU              : {torch.cuda.get_device_name(device)}")
    print(f"Extension        : {pointnet2_utils.extension_path()}")
    print(f"Anchors          : {args.anchors}")
    print(f"Tracks           : {args.tracks}")
    print(f"dt               : {args.dt:.9f} s")

    ops = CudaAnchorTemporalOps(
        softmax_sigma_m=args.sigma,
        local_radius_sigma=args.radius_sigma,
        local_hash_size_factor=args.hash_size_factor,
        global_same_track_fallback=False,
    )
    queries, sources, values3, values6, query_ids, source_ids = make_scene(args, device)

    print("\nCorrectness")
    print("-" * 96)
    reference3, reference_support = strict_local_transport_reference(
        queries, sources, values3, query_ids, source_ids, args.sigma, args.radius_sigma
    )
    result3 = ops.transport(
        query_points=queries,
        source_points=sources,
        source_values=values3,
        query_track_ids=query_ids,
        source_track_ids=source_ids,
    )
    assert_close("3-channel strict transport", result3.values, reference3, args)
    if not torch.equal(result3.support_counts, reference_support):
        raise AssertionError("3-channel support counts do not match reference")
    print("[PASS] strict transport support counts")

    reference6, reference_support6 = strict_local_transport_reference(
        queries, sources, values6, query_ids, source_ids, args.sigma, args.radius_sigma
    )
    result6 = ops.transport(
        query_points=queries,
        source_points=sources,
        source_values=values6,
        query_track_ids=query_ids,
        source_track_ids=source_ids,
    )
    assert_close("6-channel strict transport", result6.values, reference6, args)
    if not torch.equal(result6.support_counts, reference_support6):
        raise AssertionError("6-channel support counts do not match reference")
    print("[PASS] 6-channel support counts")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    current_flow = 0.01 * torch.randn((args.anchors, 3), generator=generator, device=device)
    support = torch.ones((args.anchors,), device=device, dtype=torch.int32)
    support[-8:] = 0

    previous_state6 = torch.zeros((args.anchors, 6), device=device, dtype=torch.float32)
    previous_state6[:, :3] = 0.4 * torch.randn((args.anchors, 3), generator=generator, device=device)
    previous_state6[:, 3:6] = 0.09

    kf_kwargs = dict(
        process_velocity_std_mps=0.05,
        measurement_noise_std_mps=0.10,
        initial_velocity_std_mps=0.30,
        min_innovation_variance=1.0e-6,
    )
    kf = ops.kalman_update(
        current_flow=current_flow,
        transported_previous_state6=previous_state6,
        support_counts=support,
        dt_s=args.dt,
        **kf_kwargs,
    )
    kf_ref = kf_reference(
        current_flow, previous_state6, support, args.dt, **kf_kwargs
    )
    assert_close("Velocity-only Kalman state update", kf.state6, kf_ref, args)

    # Q is specified directly in velocity space per update. Changing dt while
    # preserving the same velocity observation must therefore not change the KF
    # state. This guards against accidentally reintroducing a hidden dt^2 /
    # acceleration parameterization into the process covariance.
    alternate_dt = args.dt * 1.7
    same_velocity_flow = (current_flow / args.dt) * alternate_dt
    kf_alternate_dt = ops.kalman_update(
        current_flow=same_velocity_flow,
        transported_previous_state6=previous_state6,
        support_counts=support,
        dt_s=alternate_dt,
        **kf_kwargs,
    )
    assert_close(
        "velocity-domain Q is dt-independent per update",
        kf_alternate_dt.state6,
        kf.state6,
        args,
    )

    # A large but finite innovation is still fused by the standard KF. This
    # explicitly guards against accidentally reintroducing statistical gating.
    spike_flow = current_flow.clone()
    spike_flow[0] = torch.tensor([25.0, -20.0, 15.0], device=device) * args.dt
    spike = ops.kalman_update(
        current_flow=spike_flow,
        transported_previous_state6=previous_state6,
        support_counts=support,
        dt_s=args.dt,
        **kf_kwargs,
    )
    spike_ref = kf_reference(
        spike_flow, previous_state6, support, args.dt, **kf_kwargs
    )
    assert_close("large finite innovation uses standard KF update", spike.state6, spike_ref, args)

    # Non-finite history must be contained locally and reinitialized from the
    # current finite measurement instead of entering recursive state history.
    corrupt_history = previous_state6.clone()
    corrupt_history[0, 0] = float("nan")
    corrupt_history[1, 4] = float("nan")
    repaired = ops.kalman_update(
        current_flow=current_flow,
        transported_previous_state6=corrupt_history,
        support_counts=support,
        dt_s=args.dt,
        **kf_kwargs,
    )
    if not bool(torch.isfinite(repaired.state6).all().item()):
        raise AssertionError("Non-finite transported history escaped KF containment")
    expected_reinit = current_flow[:2] / args.dt
    assert_close("non-finite history local reinitialization", repaired.velocity[:2], expected_reinit, args)

    # Non-finite measurements cannot enter the KF arithmetic; with valid
    # history, the finite random-walk prediction is preserved.
    invalid_flow = current_flow.clone()
    invalid_flow[2, 0] = float("nan")
    invalid_measurement = ops.kalman_update(
        current_flow=invalid_flow,
        transported_previous_state6=previous_state6,
        support_counts=support,
        dt_s=args.dt,
        **kf_kwargs,
    )
    if not bool(torch.isfinite(invalid_measurement.state6[2]).all().item()):
        raise AssertionError("Non-finite measurement produced non-finite KF state")
    print("[PASS] non-finite measurement containment")

    # Long-recursion stress test. Use noisy interval velocities plus periodic
    # large finite innovations and require finite positive covariance throughout.
    recursive_count = min(args.anchors, 512)
    recursive_support = torch.ones(
        (recursive_count,), device=device, dtype=torch.int32
    )
    true_velocity = 0.3 * torch.randn(
        (recursive_count, 3), generator=generator, device=device
    )
    velocity_step = 0.02 * torch.randn(
        (recursive_count, 3), generator=generator, device=device
    )
    initial_measurement = true_velocity
    initial_flow = initial_measurement * args.dt
    recursive_zero_state = torch.zeros(
        (recursive_count, 6), device=device, dtype=torch.float32
    )
    recursive_zero_support = torch.zeros(
        (recursive_count,), device=device, dtype=torch.int32
    )
    recursive_kwargs = dict(
        process_velocity_std_mps=0.05,
        measurement_noise_std_mps=0.30,
        initial_velocity_std_mps=0.30,
        min_innovation_variance=1.0e-6,
    )
    recursive_state = ops.kalman_update(
        current_flow=initial_flow,
        transported_previous_state6=recursive_zero_state,
        support_counts=recursive_zero_support,
        dt_s=args.dt,
        **recursive_kwargs,
    ).state6
    for step in range(args.recursive_steps):
        true_velocity = true_velocity + velocity_step
        z_true = true_velocity
        measurement = z_true + 0.30 * torch.randn(
            (recursive_count, 3), generator=generator, device=device
        )
        if step % 37 == 0:
            spike_count = min(8, recursive_count)
            measurement[:spike_count] += torch.tensor(
                [12.0, -10.0, 8.0], device=device
            )
        recursive_result = ops.kalman_update(
            current_flow=measurement * args.dt,
            transported_previous_state6=recursive_state,
            support_counts=recursive_support,
            dt_s=args.dt,
            **recursive_kwargs,
        )
        recursive_state = recursive_result.state6
        if not bool(torch.isfinite(recursive_state).all().item()):
            raise AssertionError(f"Recursive KF produced non-finite state at step {step}")
        min_variance = float(recursive_state[:, 3:6].min().item())
        if min_variance <= 0.0:
            raise AssertionError(
                f"Recursive KF covariance lost positivity at step {step}: "
                f"min variance={min_variance:.6g}"
            )

    print(f"[PASS] recursive velocity-Kalman finite/positive-covariance stability ({args.recursive_steps} steps)")

    print("\nLatency")
    print("-" * 96)
    transport3_stats = benchmark_cuda(
        lambda: ops.transport(
            query_points=queries,
            source_points=sources,
            source_values=values3,
            query_track_ids=query_ids,
            source_track_ids=source_ids,
        ),
        args.warmup,
        args.iterations,
    )
    transport6_stats = benchmark_cuda(
        lambda: ops.transport(
            query_points=queries,
            source_points=sources,
            source_values=values6,
            query_track_ids=query_ids,
            source_track_ids=source_ids,
        ),
        args.warmup,
        args.iterations,
    )
    kf_stats = benchmark_cuda(
        lambda: ops.kalman_update(
            current_flow=current_flow,
            transported_previous_state6=previous_state6,
            support_counts=support,
            dt_s=args.dt,
            **kf_kwargs,
        ),
        args.warmup,
        args.iterations,
    )
    print_timing("transport C=3", transport3_stats)
    print_timing("transport C=6", transport6_stats)
    print_timing("velocity KF fused kernel", kf_stats)

    print("\nAll anchor temporal CUDA checks passed.")


if __name__ == "__main__":
    main()
