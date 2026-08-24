"""High-level world-space streaming inference adapter."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from difflow3d.model import PointConvBidirection
from .checkpoint import CheckpointReport, load_checkpoint
from .runner import DifFlow3DStreamingCudaGraphRunner, configure_fast_inference


@dataclass(frozen=True)
class DifFlow3DConfig:
    checkpoint_path: Path
    enable_tf32: bool = True
    cuda_graph_warmup: int = 10
    device: str = "cuda:0"
    num_points: int = 2048
    iters: int = 4
    coarse_iters: int | None = None
    middle_iters: int | None = None
    fine_iters: int | None = None
    uncertainty: float = 0.2
    strict_checkpoint: bool = True
    disable_bn_running_stats: bool = True
    frame_dt_s: float = 1.0 / 30.0
    max_frame_gap_s: float | None = None
    second_base_voxel_size_m: float = 0.010
    second_candidate_ratio: float = 2.5
    auto_spatial_scale: bool = True
    target_model_volume: float = 1.0
    fixed_spatial_scale: float = 1.0
    final_selection: str = "fps"
    outlier_filter_enabled: bool = False
    outlier_filter_min_component_size_ratio: float = 0.05
    enable_profiling: bool = False
    validate_finite: bool = False


@dataclass(frozen=True)
class DifFlow3DEstimate:
    source_points: torch.Tensor
    target_points: torch.Tensor
    warped_points: torch.Tensor
    residual_flow: torch.Tensor
    velocity: torch.Tensor
    valid_indices: torch.Tensor
    source_timestamp_s: float
    target_timestamp_s: float
    source_preprocess_info: dict[str, object]
    target_preprocess_info: dict[str, object]


class DifFlow3DInference:
    required_frames = 2

    def __init__(self, config: DifFlow3DConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.num_points = int(config.num_points)
        self._cached_target_timestamp_s: float | None = None
        if self.device.type != "cuda":
            raise ValueError("DifFlow3D requires a CUDA device.")

        configure_fast_inference(config.enable_tf32)
        self.model = PointConvBidirection(
            iters=config.iters,
            coarse_iters=config.coarse_iters,
            middle_iters=config.middle_iters,
            fine_iters=config.fine_iters,
        )
        self.checkpoint_report: CheckpointReport = load_checkpoint(
            self.model,
            config.checkpoint_path,
            strict=config.strict_checkpoint,
        )
        self.model.to(self.device).eval()
        if config.disable_bn_running_stats:
            for layer in self.model.modules():
                if isinstance(
                    layer,
                    (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d),
                ):
                    layer.track_running_stats = False

        self.runner = DifFlow3DStreamingCudaGraphRunner(
            self.model,
            batch_size=1,
            num_points=self.num_points,
            uncertainty=config.uncertainty,
            warmup=config.cuda_graph_warmup,
            enable_tf32=config.enable_tf32,
            dt_s=config.frame_dt_s,
            second_base_voxel_size_m=config.second_base_voxel_size_m,
            second_candidate_ratio=config.second_candidate_ratio,
            auto_spatial_scale=config.auto_spatial_scale,
            target_model_volume=config.target_model_volume,
            fixed_spatial_scale=config.fixed_spatial_scale,
            final_selection=config.final_selection,
            outlier_filter_enabled=config.outlier_filter_enabled,
            outlier_filter_min_component_size_ratio=(
                config.outlier_filter_min_component_size_ratio
            ),
            enable_profiling=config.enable_profiling,
            validate_finite=config.validate_finite,
        )

    def reset(self) -> None:
        self._cached_target_timestamp_s = None
        self.runner.reset()

    def calibrate(self, first_points: torch.Tensor) -> dict[str, object]:
        return self.runner.calibrate_world(first_points)

    def resolve_last_profile(
        self,
        *,
        synchronize: bool = False,
    ) -> dict[str, float]:
        return self.runner.resolve_profile_window(synchronize=synchronize)

    def infer(self, source_frame, target_frame) -> DifFlow3DEstimate:
        dt_s = float(target_frame.timestamp_s - source_frame.timestamp_s)
        if dt_s <= 0.0:
            raise ValueError(f"Non-increasing timestamps: dt={dt_s}.")
        if (
            self.config.max_frame_gap_s is not None
            and dt_s > self.config.max_frame_gap_s
        ):
            raise ValueError(
                f"Frame gap {dt_s:.6f}s exceeds "
                f"{self.config.max_frame_gap_s:.6f}s."
            )
        if not np.isclose(
            dt_s,
            self.config.frame_dt_s,
            rtol=1.0e-5,
            atol=1.0e-8,
        ):
            raise ValueError(
                "Captured CUDA Graph uses dt="
                f"{self.config.frame_dt_s:.9f}s, received {dt_s:.9f}s."
            )

        source_is_cached = (
            self._cached_target_timestamp_s is not None
            and np.isclose(
                self._cached_target_timestamp_s,
                source_frame.timestamp_s,
                rtol=0.0,
                atol=1.0e-12,
            )
        )

        self.runner.begin_profile_window()
        with torch.inference_mode():
            if not source_is_cached:
                self.runner.reset()
                # reset() clears profiling state, so re-open the window.
                self.runner.begin_profile_window()
                self.runner.stage_world(source_frame.first_downsample_points)
                if self.runner.replay_next() is not None:
                    raise RuntimeError("First streaming frame must only buffer.")

            self.runner.stage_world(target_frame.first_downsample_points)
            if self.runner.replay_next() is None:
                raise RuntimeError(
                    "Streaming decode did not produce a pair output."
                )

            predicted_flow = self.runner.flow_world()[0]
            source_points = self.runner.source_points_world()[0]
            target_points = self.runner.target_points_world()[0]
            warped_points = self.runner.warped_points_world()[0]
            source_selection = self.runner.source_selection_indices()
            valid_indices = source_frame.first_raw_indices.index_select(
                0,
                source_selection,
            )
            source_info = self.runner.source_preprocess_info()
            target_info = self.runner.target_preprocess_info()

        self._cached_target_timestamp_s = float(target_frame.timestamp_s)
        return DifFlow3DEstimate(
            source_points=source_points,
            target_points=target_points,
            warped_points=warped_points,
            residual_flow=predicted_flow,
            velocity=predicted_flow / dt_s,
            valid_indices=valid_indices,
            source_timestamp_s=float(source_frame.timestamp_s),
            target_timestamp_s=float(target_frame.timestamp_s),
            source_preprocess_info=source_info,
            target_preprocess_info=target_info,
        )
