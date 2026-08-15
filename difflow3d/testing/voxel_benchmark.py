"""Voxelized synthetic benchmark orchestration."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from difflow3d.config import voxel_namespace
from difflow3d.runtime import DifFlow3DConfig, DifFlow3DInference, SoftmaxAnchorMotionRecoverer
from .synthetic_scene import OnlineSceneFrame, OnlineSceneGenerator, make_obstacle_specs
from .first_voxel import FirstVoxelFrame, FirstVoxelPreprocessor
from .metrics import cuda_index, synchronize, summarize, print_timing_row, metric_summary, safe_mean
from .visualization import RvizPipelinePublisher

def run(config: dict) -> None:
    args = voxel_namespace(config)
    if args.sensor_hz <= 0.0:
        raise ValueError("--sensor-hz must be positive.")
    if args.all_points < 1:
        raise ValueError("--all-points must be positive.")
    if args.dimension_factor <= 0.0 or not np.isfinite(args.dimension_factor):
        raise ValueError("--dimension-factor must be positive and finite.")
    if args.voxel_resolution <= 0.0:
        raise ValueError("--voxel-resolution must be positive.")
    if args.second_candidate_ratio <= 1.0:
        raise ValueError("--second-candidate-ratio must be > 1.0.")
    if args.recovery_softmax_sigma <= 0.0:
        raise ValueError("--recovery-softmax-sigma must be positive.")
    if args.recovery_local_radius_sigma <= 0.0:
        raise ValueError("recovery.local_radius_sigma must be positive.")
    if args.recovery_local_hash_size_factor < 1.0:
        raise ValueError("recovery.local_hash_size_factor must be >= 1.0.")
    if args.sensor_noise_std < 0.0:
        raise ValueError("--sensor-noise-std must be non-negative.")
    if args.fps_points < 1024:
        raise ValueError("The optimized model requires --fps-points >= 1024.")
    if args.target_model_volume <= 0.0:
        raise ValueError("--target-model-volume must be positive.")

    dt_s = 1.0 / args.sensor_hz
    period_ms = 1000.0 * dt_s
    repo_path = args.difflow_repo.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else repo_path / "checkpoints" / "model_difflow_355_0.0114.pth"
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark requires CUDA.")
    torch.cuda.set_device(cuda_index(device))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_index(device))

    dimension_factor = float(args.dimension_factor)
    effective_voxel_resolution_m = float(args.voxel_resolution) * dimension_factor
    effective_sensor_noise_std_m = float(args.sensor_noise_std) * dimension_factor
    # Recovery is performed after DifFlow outputs are converted back to world
    # coordinates. The configured sigma is therefore used directly in metres.
    effective_recovery_sigma_m = float(args.recovery_softmax_sigma)
    specs = make_obstacle_specs(dimension_factor, bool(args.motion))

    scene_generator = OnlineSceneGenerator(
        frame_count=args.frames,
        dt_s=dt_s,
        mesh_resolution=args.mesh_resolution,
        total_points=args.all_points,
        sensor_noise_std_m=effective_sensor_noise_std_m,
        same_samples_across_frames=args.same_samples_across_frames,
        seed=args.seed,
        specs=specs,
    )
    preprocessor = FirstVoxelPreprocessor(
        raw_point_count=args.all_points,
        device=device,
        first_voxel_size_m=effective_voxel_resolution_m,
    )

    fixed_spatial_scale = (
        float(args.fixed_spatial_scale)
        if args.fixed_spatial_scale is not None
        else 1.0
    )
    auto_spatial_scale = bool(args.auto_spatial_scale)

    inference_config = DifFlow3DConfig(
        checkpoint_path=checkpoint,
        enable_tf32=not args.disable_tf32,
        cuda_graph_warmup=args.cuda_graph_warmup,
        device=args.device,
        num_points=args.fps_points,
        iters=args.difflow_iters,
        coarse_iters=args.difflow_coarse_iters,
        middle_iters=args.difflow_middle_iters,
        fine_iters=args.difflow_fine_iters,
        uncertainty=args.difflow_uncertainty,
        strict_checkpoint=not args.non_strict_checkpoint,
        disable_bn_running_stats=not args.keep_bn_running_stats,
        frame_dt_s=dt_s,
        max_frame_gap_s=2.0 * dt_s,
        second_base_voxel_size_m=effective_voxel_resolution_m,
        second_candidate_ratio=args.second_candidate_ratio,
        auto_spatial_scale=auto_spatial_scale,
        target_model_volume=args.target_model_volume,
        fixed_spatial_scale=fixed_spatial_scale,
        final_selection=args.final_selection,
        enable_profiling=args.detailed_runtime_breakdown,
        validate_finite=args.validate_finite,
    )

    synchronize(device)
    load_start = time.perf_counter()
    estimator = DifFlow3DInference(inference_config)
    synchronize(device)
    model_load_ms = 1000.0 * (time.perf_counter() - load_start)

    calibration_scene = scene_generator.frame(0)
    calibration_first = preprocessor.process(
        calibration_scene.points, calibration_scene.timestamp_s
    )
    print("Calibrating runner voxel-2 and spatial scale (untimed) ...")
    calibration = estimator.calibrate(calibration_first.first_downsample_points)
    second_report = calibration["second"]
    anchor_info = calibration["anchor_info"]

    print("=" * 104)
    print("Raw -> voxel-1 -> RUNNER[adaptive voxel-2 -> exact-count selection -> scale -> DifFlow3D] -> softmax recovery")
    print("=" * 104)
    print(f"Dimension factor:             {dimension_factor:.6f}")
    print(f"Motion enabled:               {bool(args.motion)}")
    print(f"Raw points/frame:             {args.all_points}")
    print(f"Base voxel-1 resolution:      {args.voxel_resolution:.6f} m")
    print(f"Effective voxel-1 resolution: {effective_voxel_resolution_m:.6f} m")
    print(f"Effective sensor noise std:   {effective_sensor_noise_std_m:.6f} m")
    print(f"First calibration count:      {calibration_first.first_count}")
    print(f"Second mode:                  {second_report.get('mode')}")
    print(f"Second voxel resolution:      {second_report.get('second_voxel_resolution_m')}")
    print(f"Target candidate count:       {estimator.runner.target_candidate_count}")
    print(f"Calibration candidates:       {second_report.get('candidate_count')}")
    print(f"Second candidate ratio:       {args.second_candidate_ratio:.3f}")
    print(f"Model points:                 {args.fps_points}")
    print(f"Final selection backend:      {args.final_selection}")
    print(f"Calibration selection mode:   {anchor_info.get('selection_mode')}")
    print(f"Auto spatial scale:           {auto_spatial_scale}")
    print(f"Target model volume:          {args.target_model_volume:.6f}")
    print(f"Spatial scale:                {estimator.runner.spatial_scale:.6f}")
    print(f"Anchor world extent:          {anchor_info.get('world_extent')}")
    print(f"Anchor world volume:          {anchor_info.get('world_volume'):.9f}")
    print(f"Anchor model volume:          {float(anchor_info.get('world_volume')) * estimator.runner.spatial_scale**3:.9f}")
    print(f"Recovery softmax sigma:       {effective_recovery_sigma_m:.6f} m")
    print(f"Recovery backend:             {args.recovery_backend}")
    if args.recovery_backend == "local":
        print(f"Recovery local radius:        {args.recovery_local_radius_sigma:.2f} sigma = {args.recovery_local_radius_sigma * effective_recovery_sigma_m:.6f} m")
        print(f"Recovery hash size factor:    {args.recovery_local_hash_size_factor:.2f}")
    print(f"Detailed runtime profiling:   {args.detailed_runtime_breakdown}")
    print(
        "DifFlow iterations C/M/F:    "
        f"{args.difflow_coarse_iters}/{args.difflow_middle_iters}/{args.difflow_fine_iters}"
    )
    print(f"Exact checkpoint:             {estimator.checkpoint_report.is_exact}")
    print(f"Model load time:              {model_load_ms:.1f} ms")

    recoverer = SoftmaxAnchorMotionRecoverer(
        chunk_size=args.recovery_chunk_size,
        softmax_sigma_m=effective_recovery_sigma_m,
        backend=args.recovery_backend,
        local_radius_sigma=args.recovery_local_radius_sigma,
        local_hash_size_factor=args.recovery_local_hash_size_factor,
    )

    # Warmup without changing the frozen voxel-2/spatial calibration.
    warm_source_scene = scene_generator.frame(0)
    warm_target_scene = scene_generator.frame(1)
    for _ in range(max(0, args.warmup)):
        estimator.reset()
        ws = preprocessor.process(warm_source_scene.points, warm_source_scene.timestamp_s)
        wt = preprocessor.process(warm_target_scene.points, warm_target_scene.timestamp_s)
        we = estimator.infer(ws, wt)
        _ = recoverer.recover(
            query_points=ws.first_downsample_points,
            anchor_points=we.source_points,
            anchor_flow=we.residual_flow,
            dt_s=dt_s,
        )
        synchronize(device)
    estimator.reset()
    torch.cuda.reset_peak_memory_stats(cuda_index(device))

    rviz = (
        RvizPipelinePublisher(
            frame_id=args.rviz_frame_id,
            max_arrows=args.rviz_max_arrows,
            vector_scale=args.rviz_vector_scale,
            cloud_max_points=args.rviz_cloud_max_points,
        ) if args.rviz else None
    )

    timing: dict[str, list[float]] = {
        "scene_generation_ms": [],
        "host_stage_ms": [],
        "h2d_ms": [],
        "first_downsample_ms": [],
        "runner_voxel2_ms": [],
        "runner_final_selection_ms": [],
        "runner_stage_scale_ms": [],
        "runner_encode_ms": [],
        "runner_decode_ms": [],
        "runner_profiled_ms": [],
        "runner_model_total_ms": [],
        "recovery_ms": [],
        "overall_wall_ms": [],
    }
    first_counts: list[int] = []
    candidate_counts: list[int] = []
    anchor_flow_epe_chunks: list[np.ndarray] = []
    first_flow_epe_chunks: list[np.ndarray] = []
    anchor_velocity_epe_chunks: list[np.ndarray] = []
    first_velocity_epe_chunks: list[np.ndarray] = []
    per_frame: list[dict[str, object]] = []
    local_neighbor_frame_stats: list[dict[str, float]] = []

    previous_prepared: FirstVoxelFrame | None = None
    previous_scene: OnlineSceneFrame | None = None
    stream_wall_start = time.perf_counter()

    print("\nStreaming frames")
    print("-" * 104)
    for target_index in range(args.frames):
        if args.realtime:
            release_time = stream_wall_start + target_index * dt_s
            sleep_s = release_time - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        scene_start = time.perf_counter()
        scene_frame = scene_generator.frame(target_index)
        scene_generation_ms = 1000.0 * (time.perf_counter() - scene_start)
        timing["scene_generation_ms"].append(scene_generation_ms)

        cycle_start = time.perf_counter()
        prepared = preprocessor.process(scene_frame.points, scene_frame.timestamp_s)
        first_counts.append(prepared.first_count)
        timing["host_stage_ms"].append(prepared.host_stage_ms)
        timing["h2d_ms"].append(prepared.h2d_ms)
        timing["first_downsample_ms"].append(prepared.first_downsample_ms)

        if previous_prepared is None or previous_scene is None:
            previous_prepared = prepared
            previous_scene = scene_frame
            if rviz is not None:
                rviz.publish_buffered(prepared)
            print(
                f"frame {target_index:03d}: raw={prepared.raw_count:7d} -> "
                f"first={prepared.first_count:7d} buffering"
            )
            continue

        model_start = torch.cuda.Event(enable_timing=True)
        model_end = torch.cuda.Event(enable_timing=True)
        recovery_start = torch.cuda.Event(enable_timing=True)
        recovery_end = torch.cuda.Event(enable_timing=True)
        model_start.record()
        estimate = estimator.infer(previous_prepared, prepared)
        model_end.record()
        recovery_start.record()
        recovery = recoverer.recover(
            query_points=previous_prepared.first_downsample_points,
            anchor_points=estimate.source_points,
            anchor_flow=estimate.residual_flow,
            dt_s=dt_s,
        )
        recovery_end.record()
        recovery_end.synchronize()

        runner_model_total_ms = float(model_start.elapsed_time(model_end))
        recovery_ms = float(recovery_start.elapsed_time(recovery_end))
        overall_wall_ms = 1000.0 * (time.perf_counter() - cycle_start)

        local_stats = None
        if recovery.local_neighbor_counts is not None and args.recovery_report_local_stats:
            counts = recovery.local_neighbor_counts
            counts_f = counts.float()
            # Diagnostics are intentionally outside the timed recovery/cycle.
            local_stats = {
                "mean": float(counts_f.mean().item()),
                "median": float(counts_f.median().item()),
                "p95": float(torch.quantile(counts_f, 0.95).item()),
                "max": float(counts.max().item()),
                "empty_ratio": float((counts == 0).float().mean().item()),
            }
            local_neighbor_frame_stats.append(local_stats)

        profile = estimator.resolve_last_profile(synchronize=False)
        target_info = estimate.target_preprocess_info
        voxel2_ms = float(profile.get("voxel2_ms", 0.0))
        selection_ms = float(profile.get("final_selection_ms", 0.0))
        stage_scale_ms = float(profile.get("stage_scale_ms", 0.0))
        encode_ms = float(profile.get("encode_ms", 0.0))
        decode_ms = float(profile.get("decode_ms", 0.0))
        profiled_runner_ms = float(profile.get("profiled_runner_ms", 0.0))
        candidate_count = int(target_info.get("candidate_count", args.fps_points))
        candidate_counts.append(candidate_count)
        timing["runner_voxel2_ms"].append(voxel2_ms)
        timing["runner_final_selection_ms"].append(selection_ms)
        timing["runner_stage_scale_ms"].append(stage_scale_ms)
        timing["runner_encode_ms"].append(encode_ms)
        timing["runner_decode_ms"].append(decode_ms)
        timing["runner_profiled_ms"].append(profiled_runner_ms)
        timing["runner_model_total_ms"].append(runner_model_total_ms)
        timing["recovery_ms"].append(recovery_ms)
        timing["overall_wall_ms"].append(overall_wall_ms)

        anchor_raw_indices = estimate.valid_indices.detach().cpu().numpy().astype(np.int64)
        first_raw_indices = previous_prepared.first_raw_indices.detach().cpu().numpy().astype(np.int64)
        anchor_pred_flow = estimate.residual_flow.detach().cpu().numpy()
        anchor_pred_vel = estimate.velocity.detach().cpu().numpy()
        first_pred_flow = recovery.flow.detach().cpu().numpy()
        first_pred_vel = recovery.velocity.detach().cpu().numpy()

        anchor_gt_flow = previous_scene.gt_flow_to_next[anchor_raw_indices]
        first_gt_flow = previous_scene.gt_flow_to_next[first_raw_indices]
        anchor_gt_vel = anchor_gt_flow / np.float32(dt_s)
        first_gt_vel = first_gt_flow / np.float32(dt_s)

        anchor_flow_epe = np.linalg.norm(anchor_pred_flow - anchor_gt_flow, axis=1)
        first_flow_epe = np.linalg.norm(first_pred_flow - first_gt_flow, axis=1)
        anchor_vel_epe = np.linalg.norm(anchor_pred_vel - anchor_gt_vel, axis=1)
        first_vel_epe = np.linalg.norm(first_pred_vel - first_gt_vel, axis=1)
        anchor_flow_epe_chunks.append(anchor_flow_epe)
        first_flow_epe_chunks.append(first_flow_epe)
        anchor_velocity_epe_chunks.append(anchor_vel_epe)
        first_velocity_epe_chunks.append(first_vel_epe)

        row = {
            "source_index": target_index - 1,
            "target_index": target_index,
            "raw_points": prepared.raw_count,
            "first_points": prepared.first_count,
            "candidate_points": candidate_count,
            "fps_points": args.fps_points,
            "final_selection": args.final_selection,
            "selection_mode": target_info.get("selection_mode"),
            "second_voxel_used": bool(target_info.get("second_voxel_used", False)),
            "spatial_scale": float(target_info.get("spatial_scale", estimator.runner.spatial_scale)),
            "host_stage_ms": prepared.host_stage_ms,
            "h2d_ms": prepared.h2d_ms,
            "first_downsample_ms": prepared.first_downsample_ms,
            "runner_voxel2_ms": voxel2_ms,
            "runner_final_selection_ms": selection_ms,
            "runner_stage_scale_ms": stage_scale_ms,
            "runner_encode_ms": encode_ms,
            "runner_decode_ms": decode_ms,
            "runner_profiled_ms": profiled_runner_ms,
            "runner_model_total_ms": runner_model_total_ms,
            "recovery_ms": recovery_ms,
            "overall_wall_ms": overall_wall_ms,
            "anchor_mean_epe_m": float(anchor_flow_epe.mean()),
            "first_mean_epe_m": float(first_flow_epe.mean()),
            "deadline_miss": bool(overall_wall_ms > period_ms),
        }
        if local_stats is not None:
            row["local_recovery"] = local_stats
        per_frame.append(row)

        print(
            f"{target_index-1:03d}->{target_index:03d} | "
            f"raw {prepared.raw_count:7d} -> first {prepared.first_count:6d} -> "
            f"cand {candidate_count:5d} -> {args.fps_points:4d} "
            f"[{target_info.get('selection_mode')}] | "
            f"V1 {prepared.first_downsample_ms:5.2f}  "
            f"V2 {voxel2_ms:5.2f}  select {selection_ms:5.2f}  "
            f"stage {stage_scale_ms:5.2f} enc {encode_ms:5.2f} dec {decode_ms:5.2f}  "
            f"runner {runner_model_total_ms:6.2f}  recover {recovery_ms:6.2f}  "
            f"overall {overall_wall_ms:7.2f} ms | "
            f"scale {estimator.runner.spatial_scale:5.2f} | "
            f"anchor EPE {anchor_flow_epe.mean():.5f}  first EPE {first_flow_epe.mean():.5f} m"
        )

        if rviz is not None and target_index % args.rviz_publish_every == 0:
            rviz.publish_pair(
                source_frame=previous_prepared,
                target_frame=prepared,
                estimate=estimate,
                recovery=recovery,
                anchor_gt_flow=anchor_gt_flow,
                first_gt_flow=first_gt_flow,
            )

        previous_prepared = prepared
        previous_scene = scene_frame

    if not timing["overall_wall_ms"]:
        raise RuntimeError("No DifFlow3D pair was produced.")

    timing_summary = {key: summarize(values) for key, values in timing.items()}
    first_array = np.asarray(first_counts, dtype=np.float64)
    candidate_array = np.asarray(candidate_counts, dtype=np.float64)
    overall_array = np.asarray(timing["overall_wall_ms"], dtype=np.float64)
    anchor_flow_all = np.concatenate(anchor_flow_epe_chunks)
    first_flow_all = np.concatenate(first_flow_epe_chunks)
    anchor_vel_all = np.concatenate(anchor_velocity_epe_chunks)
    first_vel_all = np.concatenate(first_velocity_epe_chunks)

    print("\n" + "=" * 104)
    print("Point-count / scale summary")
    print("=" * 104)
    print(f"Raw points:                    {args.all_points}")
    print(f"First points mean/median:      {first_array.mean():.1f} / {np.median(first_array):.1f}")
    print(f"Candidate mean/median:         {candidate_array.mean():.1f} / {np.median(candidate_array):.1f}")
    print(f"Final anchors:                 {args.fps_points}")
    print(f"Frozen spatial scale:          {estimator.runner.spatial_scale:.6f}")
    print(f"Second voxel resolution:       {estimator.runner.second_voxel_size_m}")
    if local_neighbor_frame_stats:
        local_mean = np.mean([x["mean"] for x in local_neighbor_frame_stats])
        local_median = np.median([x["median"] for x in local_neighbor_frame_stats])
        local_p95 = np.mean([x["p95"] for x in local_neighbor_frame_stats])
        local_max = np.max([x["max"] for x in local_neighbor_frame_stats])
        local_empty = np.mean([x["empty_ratio"] for x in local_neighbor_frame_stats])
        print(f"Local recovery neighbors:      mean={local_mean:.2f} median={local_median:.1f} p95~={local_p95:.1f} max={local_max:.0f}")
        print(f"Local recovery fallback ratio: {local_empty:.6f}")

    print("\n" + "=" * 104)
    print("Cycle-time statistics")
    print("=" * 104)
    for key, label in (
        ("host_stage_ms", "CPU -> pinned stage"),
        ("h2d_ms", "Pinned H2D"),
        ("first_downsample_ms", "GPU first voxel"),
        ("runner_voxel2_ms", "Runner voxel-2"),
        ("runner_final_selection_ms", "Runner final selection"),
        ("runner_stage_scale_ms", "Runner stage + scale"),
        ("runner_encode_ms", "Runner encode"),
        ("runner_decode_ms", "Runner decode"),
        ("runner_profiled_ms", "Runner profiled sum"),
        ("runner_model_total_ms", "Runner preprocess + DifFlow"),
        ("recovery_ms", "Dense recovery"),
        ("overall_wall_ms", "Online overall"),
    ):
        print_timing_row(label, timing_summary[key])

    print("\n" + "=" * 104)
    print("Accuracy")
    print("=" * 104)
    print(f"Anchor flow EPE mean/median/P95: {anchor_flow_all.mean():.6f} / {np.median(anchor_flow_all):.6f} / {np.percentile(anchor_flow_all,95):.6f} m")
    print(f"First flow EPE mean/median/P95:  {first_flow_all.mean():.6f} / {np.median(first_flow_all):.6f} / {np.percentile(first_flow_all,95):.6f} m")
    print(f"Anchor velocity EPE mean/P95:    {anchor_vel_all.mean():.6f} / {np.percentile(anchor_vel_all,95):.6f} m/s")
    print(f"First velocity EPE mean/P95:     {first_vel_all.mean():.6f} / {np.percentile(first_vel_all,95):.6f} m/s")
    print(f"Deadline miss ratio:              {np.mean(overall_array > period_ms):.3f}")
    print(f"Sustainable Hz from median:       {1000.0 / np.median(overall_array):.2f}")

    if args.json_output is not None:
        result = {
            "dimension_factor": dimension_factor,
            "motion": bool(args.motion),
            "effective_voxel_resolution_m": effective_voxel_resolution_m,
            "effective_sensor_noise_std_m": effective_sensor_noise_std_m,
            "effective_recovery_softmax_sigma_m": effective_recovery_sigma_m,
            "fps_points": int(args.fps_points),
            "final_selection": args.final_selection,
            "iterations": {
                "coarse": int(args.difflow_coarse_iters),
                "middle": int(args.difflow_middle_iters),
                "fine": int(args.difflow_fine_iters),
            },
            "second_candidate_ratio": float(args.second_candidate_ratio),
            "recovery_backend": args.recovery_backend,
            "recovery_local_radius_sigma": float(args.recovery_local_radius_sigma),
            "recovery_local_hash_size_factor": float(args.recovery_local_hash_size_factor),
            "recovery_report_local_stats": bool(args.recovery_report_local_stats),
            "target_model_volume": float(args.target_model_volume),
            "calibration": calibration,
            "spatial_scale": estimator.runner.spatial_scale,
            "second_voxel_resolution_m": estimator.runner.second_voxel_size_m,
            "timing_ms": timing_summary,
            "accuracy": {
                "anchor_flow_epe_m": metric_summary(anchor_flow_all),
                "first_flow_epe_m": metric_summary(first_flow_all),
                "anchor_velocity_epe_mps": metric_summary(anchor_vel_all),
                "first_velocity_epe_mps": metric_summary(first_vel_all),
            },
            "local_recovery_frame_stats": local_neighbor_frame_stats,
            "per_frame": per_frame,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2))
        print(f"Wrote JSON: {args.json_output}")

    if rviz is not None:
        rviz.hold(args.rviz_hold_seconds)
        rviz.close()
