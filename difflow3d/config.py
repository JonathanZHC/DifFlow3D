"""YAML configuration helpers for inference and benchmark scripts."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import yaml


_ITERATION_LEVELS = ("coarse", "middle", "fine")


def load_config(path: str | Path) -> dict:
    """Load one repository config and attach resolved repository metadata."""
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    data["_config_path"] = str(path)
    data["_repo_root"] = str(path.parent.parent.resolve())
    return data


def resolve_repo_path(config: dict, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(config["_repo_root"]) / path


def parse_iteration_schedule(value) -> dict[str, int]:
    """Normalize scalar or coarse/middle/fine recurrent iteration counts."""
    if isinstance(value, bool):
        raise ValueError("model.iterations must be an integer or mapping, not bool.")

    if isinstance(value, int):
        schedule = {level: int(value) for level in _ITERATION_LEVELS}
    elif isinstance(value, dict):
        missing = set(_ITERATION_LEVELS) - set(value)
        if missing:
            raise ValueError(
                "model.iterations must define coarse/middle/fine; "
                f"missing {sorted(missing)}."
            )
        schedule = {level: int(value[level]) for level in _ITERATION_LEVELS}
    else:
        raise ValueError("model.iterations must be an integer or mapping.")

    if min(schedule.values()) < 1:
        raise ValueError("All model.iterations values must be >= 1.")
    return schedule


def parse_spatial_scale_config(preprocessing: dict) -> dict[str, bool | float]:
    """Validate and normalize preprocessing.auto_spatial_scale."""
    value = preprocessing.get("auto_spatial_scale")
    if not isinstance(value, dict):
        raise ValueError(
            "preprocessing.auto_spatial_scale must be a mapping with "
            "enable, target_model_volume, and fixed_spatial_scale."
        )

    required = {
        "enable",
        "target_model_volume",
        "fixed_spatial_scale",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            "preprocessing.auto_spatial_scale is missing "
            f"{sorted(missing)}."
        )
    if not isinstance(value["enable"], bool):
        raise ValueError(
            "preprocessing.auto_spatial_scale.enable must be a boolean."
        )

    target_model_volume = float(value["target_model_volume"])
    fixed_spatial_scale = float(value["fixed_spatial_scale"])
    if not math.isfinite(target_model_volume) or target_model_volume <= 0.0:
        raise ValueError(
            "preprocessing.auto_spatial_scale.target_model_volume must be "
            "positive."
        )
    if not math.isfinite(fixed_spatial_scale) or fixed_spatial_scale <= 0.0:
        raise ValueError(
            "preprocessing.auto_spatial_scale.fixed_spatial_scale must be "
            "positive."
        )

    return {
        "enable": value["enable"],
        "target_model_volume": target_model_volume,
        "fixed_spatial_scale": fixed_spatial_scale,
    }


def voxel_namespace(config: dict) -> SimpleNamespace:
    """Flatten the YAML sections used by the synthetic voxel benchmark."""
    root = Path(config["_repo_root"])
    runtime = config["runtime"]
    model = config["model"]
    prep = config["preprocessing"]
    recovery = config["recovery"]
    benchmark = config["benchmark"]
    rviz = config["rviz"]
    profiling = config.get("profiling", {})
    outlier = prep.get("outlier_filter", {})
    spatial_scale = parse_spatial_scale_config(prep)
    iterations = parse_iteration_schedule(model["iterations"])

    return SimpleNamespace(
        difflow_repo=root,
        checkpoint=resolve_repo_path(config, model["checkpoint"]),
        device=runtime["device"],
        disable_tf32=not bool(runtime["enable_tf32"]),
        cuda_graph_warmup=int(runtime["cuda_graph_warmup"]),
        validate_finite=bool(runtime.get("validate_finite", False)),
        detailed_runtime_breakdown=bool(
            profiling.get("detailed_runtime_breakdown", False)
        ),
        fps_points=int(prep["fps_points"]),
        final_selection=str(prep.get("final_selection", "uniform")),
        difflow_iters=max(iterations.values()),  # legacy constructor fallback
        difflow_coarse_iters=iterations["coarse"],
        difflow_middle_iters=iterations["middle"],
        difflow_fine_iters=iterations["fine"],
        difflow_uncertainty=float(model["uncertainty"]),
        non_strict_checkpoint=not bool(model["strict_checkpoint"]),
        keep_bn_running_stats=not bool(model["disable_bn_running_stats"]),
        second_candidate_ratio=float(prep["second_candidate_ratio"]),
        outlier_filter_enabled=bool(outlier.get("enabled", False)),
        outlier_filter_tiny_component_max_voxels=int(
            outlier.get("tiny_component_max_voxels", 2)
        ),
        outlier_filter_max_small_component_fraction=float(
            outlier.get("max_small_component_fraction", 0.005)
        ),
        outlier_filter_support_radius_voxels=int(
            outlier.get("support_radius_voxels", 1)
        ),
        outlier_filter_min_supported_fraction=float(
            outlier.get("min_supported_fraction", 0.3)
        ),
        auto_spatial_scale=bool(spatial_scale["enable"]),
        fixed_spatial_scale=float(spatial_scale["fixed_spatial_scale"]),
        target_model_volume=float(spatial_scale["target_model_volume"]),
        recovery_backend=str(recovery.get("backend", "local")),
        recovery_chunk_size=int(recovery["chunk_size"]),
        recovery_softmax_sigma=float(recovery["softmax_sigma_m"]),
        recovery_local_radius_sigma=float(
            recovery.get("local_radius_sigma", 4.0)
        ),
        recovery_local_hash_size_factor=float(
            recovery.get("local_hash_size_factor", 4.0)
        ),
        recovery_report_local_stats=bool(
            recovery.get("report_local_stats", False)
        ),
        dimension_factor=float(benchmark["dimension_factor"]),
        motion=bool(benchmark["motion"]),
        all_points=int(benchmark["all_points"]),
        voxel_resolution=float(benchmark["voxel_resolution_m"]),
        sensor_noise_std=float(benchmark["sensor_noise_std_m"]),
        frames=int(benchmark["frames"]),
        sensor_hz=float(benchmark["sensor_hz"]),
        warmup=int(benchmark["warmup"]),
        mesh_resolution=int(benchmark["mesh_resolution"]),
        same_samples_across_frames=bool(
            benchmark["same_samples_across_frames"]
        ),
        seed=int(benchmark["seed"]),
        realtime=bool(benchmark["realtime"]),
        json_output=resolve_repo_path(config, benchmark.get("json_output")),
        rviz=bool(rviz["enabled"]),
        rviz_frame_id=rviz["frame_id"],
        rviz_max_arrows=int(rviz["max_arrows"]),
        rviz_vector_scale=float(rviz["vector_scale"]),
        rviz_cloud_max_points=int(rviz["cloud_max_points"]),
        rviz_publish_every=int(rviz["publish_every"]),
        rviz_hold_seconds=float(rviz["hold_seconds"]),
    )
