"""Direct synthetic superquadric test using the production world-space runner."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import torch

from difflow3d.config import (
    parse_iteration_schedule,
    parse_spatial_scale_config,
    resolve_repo_path,
)
from difflow3d.runtime import DifFlow3DConfig, DifFlow3DInference
from difflow3d.runtime.voxel_outlier import resolve_voxel_outlier_statistics
from .first_voxel import FirstVoxelFrame
from .metrics import cuda_index, synchronize, summarize, print_timing_row
from .synthetic_scene import OnlineSceneGenerator, make_obstacle_specs


def _as_direct_frame(scene_frame, device: torch.device) -> FirstVoxelFrame:
    points = torch.from_numpy(np.ascontiguousarray(scene_frame.points, dtype=np.float32)).to(device, non_blocking=True)
    indices = torch.arange(points.shape[0], device=device, dtype=torch.long)
    return FirstVoxelFrame(
        raw_points_cpu=scene_frame.points,
        first_downsample_points=points,
        first_raw_indices=indices,
        timestamp_s=float(scene_frame.timestamp_s),
        raw_count=int(points.shape[0]),
        first_count=int(points.shape[0]),
        host_stage_ms=0.0,
        h2d_ms=0.0,
        first_downsample_ms=0.0,
        preprocess_gpu_ms=0.0,
        preprocess_wall_ms=0.0,
    )


def run(config: dict) -> None:
    runtime = config["runtime"]
    model_cfg = config["model"]
    prep_cfg = config["preprocessing"]
    outlier_cfg = prep_cfg.get("outlier_filter", {})
    spatial_scale_cfg = parse_spatial_scale_config(prep_cfg)
    base_bench = config["benchmark"]
    bench = config["superquadrics_benchmark"]

    dimension_factor = float(base_bench["dimension_factor"])
    motion = bool(base_bench["motion"])
    fps_points = int(prep_cfg["fps_points"])
    if fps_points < 1024:
        raise ValueError("preprocessing.fps_points must be >= 1024.")
    sensor_hz = float(bench["sensor_hz"])
    if sensor_hz <= 0.0:
        raise ValueError("superquadrics_benchmark.sensor_hz must be positive.")
    dt_s = 1.0 / sensor_hz
    device = torch.device(runtime["device"])
    if device.type != "cuda":
        raise ValueError("This benchmark requires CUDA.")
    torch.cuda.set_device(cuda_index(device))

    specs = make_obstacle_specs(dimension_factor, motion)
    base_points = sum(spec.points_per_frame for spec in specs)
    total_points = int(base_points * int(bench["point_multiplier"]))
    effective_noise = float(base_bench["sensor_noise_std_m"]) * dimension_factor
    scene = OnlineSceneGenerator(
        frame_count=int(bench["frames"]),
        dt_s=dt_s,
        mesh_resolution=int(bench["mesh_resolution"]),
        total_points=total_points,
        sensor_noise_std_m=effective_noise,
        same_samples_across_frames=bool(bench["same_samples_across_frames"]),
        seed=int(bench["seed"]),
        specs=specs,
    )

    checkpoint = resolve_repo_path(config, model_cfg["checkpoint"])
    iterations = parse_iteration_schedule(model_cfg["iterations"])
    inference_cfg = DifFlow3DConfig(
        checkpoint_path=checkpoint,
        enable_tf32=bool(runtime["enable_tf32"]),
        cuda_graph_warmup=int(runtime["cuda_graph_warmup"]),
        device=str(runtime["device"]),
        num_points=fps_points,
        iters=max(iterations.values()),
        coarse_iters=iterations["coarse"],
        middle_iters=iterations["middle"],
        fine_iters=iterations["fine"],
        uncertainty=float(model_cfg["uncertainty"]),
        strict_checkpoint=bool(model_cfg["strict_checkpoint"]),
        disable_bn_running_stats=bool(model_cfg["disable_bn_running_stats"]),
        frame_dt_s=dt_s,
        max_frame_gap_s=2.0 * dt_s,
        second_base_voxel_size_m=float(base_bench["voxel_resolution_m"]) * dimension_factor,
        second_candidate_ratio=float(prep_cfg["second_candidate_ratio"]),
        auto_spatial_scale=bool(spatial_scale_cfg["enable"]),
        target_model_volume=float(
            spatial_scale_cfg["target_model_volume"]
        ),
        fixed_spatial_scale=float(
            spatial_scale_cfg["fixed_spatial_scale"]
        ),
        final_selection=str(prep_cfg.get("final_selection", "fps")),
        outlier_filter_enabled=bool(outlier_cfg.get("enabled", False)),
        outlier_filter_min_component_size_ratio=float(
            outlier_cfg.get("min_component_size_ratio", 0.05)
        ),
        enable_profiling=bool(
            config.get("profiling", {}).get("detailed_runtime_breakdown", False)
        ),
        validate_finite=bool(runtime.get("validate_finite", False)),
    )
    synchronize(device)
    estimator = DifFlow3DInference(inference_cfg)

    first_scene = scene.frame(0)
    first_frame = _as_direct_frame(first_scene, device)
    calibration = estimator.calibrate(first_frame.first_downsample_points)
    print("=" * 96)
    print("Direct superquadric world PCD -> runner[voxel-2/filter/select/scale] -> DifFlow3D")
    print("=" * 96)
    print(f"Dimension factor:       {dimension_factor:.6f}")
    print(f"Motion enabled:         {motion}")
    print(f"Raw points/frame:       {total_points}")
    print(f"Model points:           {fps_points}")
    print(f"Candidate ratio:        {float(prep_cfg['second_candidate_ratio']):.3f}")
    print(f"Final selection:        {str(prep_cfg.get('final_selection', 'fps'))}")
    print(f"Outlier filter:         {bool(outlier_cfg.get('enabled', False))}")
    if bool(outlier_cfg.get("enabled", False)):
        print(
            "Outlier minimum component/max ratio: "
            f"{float(outlier_cfg.get('min_component_size_ratio', 0.05)):.4f}"
        )
    print(
        "Iterations C/M/F:       "
        f"{iterations['coarse']}/{iterations['middle']}/{iterations['fine']}"
    )
    print(f"Spatial scale:          {estimator.runner.spatial_scale:.6f}")
    print(f"Selection mode:         {calibration['anchor_info']['selection_mode']}")
    print(f"Checkpoint exact:       {estimator.checkpoint_report.is_exact}")

    for _ in range(int(bench["warmup"])):
        estimator.reset()
        estimator.infer(_as_direct_frame(scene.frame(0), device), _as_direct_frame(scene.frame(1), device))
        synchronize(device)
    estimator.reset()

    model_times: list[float] = []
    flow_epes: list[float] = []
    previous_scene = None
    previous_frame = None
    start_wall = time.perf_counter()
    for index in range(int(bench["frames"])):
        if bool(bench["realtime"]):
            release = start_wall + index * dt_s
            if release > time.perf_counter():
                time.sleep(release - time.perf_counter())
        current_scene = scene.frame(index)
        current_frame = _as_direct_frame(current_scene, device)
        if previous_scene is None:
            previous_scene, previous_frame = current_scene, current_frame
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        estimate = estimator.infer(previous_frame, current_frame)
        end.record(); end.synchronize()
        model_times.append(float(start.elapsed_time(end)))
        target_info = resolve_voxel_outlier_statistics(
            estimate.target_preprocess_info
        )
        stats = target_info.get("outlier_filter_statistics")
        if bool(outlier_cfg.get("enabled", False)):
            if isinstance(stats, dict):
                print(
                    f"frame {index:03d} outlier: "
                    f"blocks={int(stats['component_count'])}, "
                    f"retained={int(stats['retained_component_count'])}, "
                    "instances_with_removals="
                    f"{int(stats['instances_with_removed_components'])}, "
                    f"removed_voxels={int(stats['removed_voxel_count'])}"
                )
            else:
                print(f"frame {index:03d} outlier: voxel-2 bypassed")
        gt = torch.from_numpy(previous_scene.gt_flow_to_next).to(device)
        gt_anchor = gt.index_select(0, estimate.valid_indices)
        epe = torch.linalg.vector_norm(estimate.residual_flow - gt_anchor, dim=1).mean().item()
        flow_epes.append(float(epe))
        previous_scene, previous_frame = current_scene, current_frame

    print("\nTiming")
    print_timing_row("Runner + DifFlow", summarize(model_times))
    if flow_epes:
        array = np.asarray(flow_epes, dtype=np.float64)
        print(f"Flow EPE mean / p95:    {array.mean():.6f} / {np.percentile(array,95):.6f} m")
