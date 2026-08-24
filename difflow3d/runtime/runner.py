"""CUDA-Graph streaming runner for production DifFlow3D inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from difflow3d.model import EncodedFrame, PointConvBidirection
from .preprocessing import AdaptivePointPreprocessor, PreprocessTimingEvents


def configure_fast_inference(enable_tf32: bool = True) -> None:
    enabled = bool(enable_tf32)
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


@dataclass(frozen=True)
class _ProfileEventPair:
    label: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class DifFlow3DStreamingCudaGraphRunner:
    """Double-buffered online CUDA Graph runner with deployment preprocessing.

    Hot-path invariants:
      * variable-size world clouds are reduced to exactly ``num_points``;
      * model coordinates use one frozen isotropic spatial scale;
      * frame encodings are reused across adjacent pairs;
      * no GPU->CPU diagnostic synchronization occurs after calibration;
      * optional detailed timing records CUDA events but does not synchronize.
    """

    def __init__(
        self,
        model: PointConvBidirection,
        *,
        batch_size: int = 1,
        num_points: int = 1024,
        uncertainty: float = 0.2,
        warmup: int = 10,
        enable_tf32: bool = True,
        dt_s: float = 1.0,
        second_base_voxel_size_m: float = 0.010,
        second_candidate_ratio: float = 2.5,
        auto_spatial_scale: bool = False,
        target_model_volume: float = 1.0,
        fixed_spatial_scale: float = 1.0,
        volume_epsilon: float = 1.0e-12,
        final_selection: str = "fps",
        outlier_filter_enabled: bool = False,
        outlier_filter_tiny_component_max_voxels: int = 2,
        outlier_filter_max_small_component_fraction: float = 0.005,
        outlier_filter_support_radius_voxels: int = 1,
        outlier_filter_min_supported_fraction: float = 0.3,
        enable_profiling: bool = False,
        validate_finite: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Streaming CUDA Graph inference requires CUDA.")
        if batch_size < 1 or num_points < 1024 or warmup < 1:
            raise ValueError(
                "The optimized streaming runner requires num_points >= 1024 "
                "and positive batch_size/warmup."
            )
        if second_base_voxel_size_m <= 0.0:
            raise ValueError("second_base_voxel_size_m must be positive.")
        if second_candidate_ratio <= 1.0:
            raise ValueError("second_candidate_ratio must be > 1.0.")
        if target_model_volume <= 0.0:
            raise ValueError("target_model_volume must be positive.")
        if fixed_spatial_scale <= 0.0:
            raise ValueError("fixed_spatial_scale must be positive.")
        if volume_epsilon <= 0.0:
            raise ValueError("volume_epsilon must be positive.")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")

        if model.training:
            model.eval()
        # recurrent0/1/2 correspond to fine/middle/coarse pyramid levels.
        fine_iterations = int(getattr(model.recurrent0, "iters", 0))
        middle_iterations = int(getattr(model.recurrent1, "iters", 0))
        coarse_iterations = int(getattr(model.recurrent2, "iters", 0))
        if min(fine_iterations, middle_iterations, coarse_iterations) < 1:
            raise ValueError(
                "All recurrent iteration counts must be positive: "
                f"coarse={coarse_iterations}, middle={middle_iterations}, "
                f"fine={fine_iterations}."
            )

        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("The model has no parameters.") from error
        if device.type != "cuda":
            raise ValueError("Move the model to CUDA before graph capture.")

        configure_fast_inference(enable_tf32)

        self.model = model
        self.device = device
        self.batch_size = int(batch_size)
        self.num_points = int(num_points)
        self.iteration_schedule = {
            "coarse": coarse_iterations,
            "middle": middle_iterations,
            "fine": fine_iterations,
        }
        self.uncertainty = float(uncertainty)
        self.dt_s = torch.tensor(float(dt_s), device=device, dtype=torch.float32)
        self.enable_profiling = bool(enable_profiling)

        self.preprocessor = AdaptivePointPreprocessor(
            num_points=self.num_points,
            second_base_voxel_size_m=second_base_voxel_size_m,
            second_candidate_ratio=second_candidate_ratio,
            auto_spatial_scale=auto_spatial_scale,
            target_model_volume=target_model_volume,
            fixed_spatial_scale=fixed_spatial_scale,
            volume_epsilon=volume_epsilon,
            final_selection=final_selection,
            outlier_filter_enabled=outlier_filter_enabled,
            outlier_filter_tiny_component_max_voxels=(
                outlier_filter_tiny_component_max_voxels
            ),
            outlier_filter_max_small_component_fraction=(
                outlier_filter_max_small_component_fraction
            ),
            outlier_filter_support_radius_voxels=(
                outlier_filter_support_radius_voxels
            ),
            outlier_filter_min_supported_fraction=(
                outlier_filter_min_supported_fraction
            ),
            enable_timing=self.enable_profiling,
            validate_finite=validate_finite,
        )

        shape = (self.batch_size, self.num_points, 3)
        self.input_a = torch.empty(shape, device=device, dtype=torch.float32)
        self.input_b = torch.empty_like(self.input_a)

        self.encode_graph_a = torch.cuda.CUDAGraph()
        self.encode_graph_b = torch.cuda.CUDAGraph()
        self.decode_graph_ab = torch.cuda.CUDAGraph()
        self.decode_graph_ba = torch.cuda.CUDAGraph()

        self.encoded_a: EncodedFrame | None = None
        self.encoded_b: EncodedFrame | None = None
        self.output_ab = None
        self.output_ba = None
        self.flow_ab: torch.Tensor | None = None
        self.flow_ba: torch.Tensor | None = None
        self.warped_ab: torch.Tensor | None = None
        self.warped_ba: torch.Tensor | None = None

        self._slot_selection_indices: list[torch.Tensor | None] = [None, None]
        self._slot_point_ids: list[torch.Tensor | None] = [None, None]
        self._slot_preprocess_info: list[dict[str, object] | None] = [None, None]

        self._next_slot = 0
        self._previous_slot: int | None = None
        self._last_source_slot: int | None = None
        self._last_target_slot: int | None = None
        self._current_output = None
        self._current_flow: torch.Tensor | None = None
        self._current_warped: torch.Tensor | None = None

        self._profile_active = False
        self._profile_events: list[_ProfileEventPair] = []
        self._last_profile: dict[str, float] | None = None

        self._capture(int(warmup))

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    def _warmup(self, count: int) -> None:
        current = torch.cuda.current_stream(self.device)
        setup = torch.cuda.Stream(device=self.device)
        setup.wait_stream(current)

        with torch.cuda.stream(setup), torch.inference_mode():
            for _ in range(count):
                encoded_a = self.model.encode_frame(self.input_a, self.input_a)
                encoded_b = self.model.encode_frame(self.input_b, self.input_b)
                output_ab = self.model.decode_pair(
                    encoded_a, encoded_b, None, self.uncertainty
                )
                output_ba = self.model.decode_pair(
                    encoded_b, encoded_a, None, self.uncertainty
                )
                _ = output_ab[0][0][0].permute(0, 2, 1).contiguous()
                _ = output_ba[0][0][0].permute(0, 2, 1).contiguous()

        current.wait_stream(setup)
        torch.cuda.synchronize(self.device)

    def _capture(self, warmup: int) -> None:
        # RNG initialization deliberately occurs outside CUDA graph capture.
        self.input_a.normal_(0.0, 0.25)
        self.input_b.normal_(0.0, 0.25)
        self._warmup(warmup)

        with torch.cuda.graph(self.encode_graph_a):
            with torch.inference_mode():
                self.encoded_a = self.model.encode_frame(
                    self.input_a,
                    self.input_a,
                )

        with torch.cuda.graph(self.encode_graph_b):
            with torch.inference_mode():
                self.encoded_b = self.model.encode_frame(
                    self.input_b,
                    self.input_b,
                )

        self.encode_graph_a.replay()
        self.encode_graph_b.replay()
        torch.cuda.synchronize(self.device)

        assert self.encoded_a is not None
        assert self.encoded_b is not None

        with torch.cuda.graph(self.decode_graph_ab):
            with torch.inference_mode():
                self.output_ab = self.model.decode_pair(
                    self.encoded_a,
                    self.encoded_b,
                    None,
                    self.uncertainty,
                )
                self.flow_ab = (
                    self.output_ab[0][0][0]
                    .permute(0, 2, 1)
                    .float()
                    .contiguous()
                )
                self.warped_ab = self.input_a + self.flow_ab

        with torch.cuda.graph(self.decode_graph_ba):
            with torch.inference_mode():
                self.output_ba = self.model.decode_pair(
                    self.encoded_b,
                    self.encoded_a,
                    None,
                    self.uncertainty,
                )
                self.flow_ba = (
                    self.output_ba[0][0][0]
                    .permute(0, 2, 1)
                    .float()
                    .contiguous()
                )
                self.warped_ba = self.input_b + self.flow_ba

        torch.cuda.synchronize(self.device)
        self.reset()

    # ------------------------------------------------------------------
    # Non-synchronizing detailed profiler
    # ------------------------------------------------------------------

    def begin_profile_window(self) -> None:
        if not self.enable_profiling:
            return
        self._profile_events = []
        self._profile_active = True
        self._last_profile = None

    def _profile_pair(
        self,
        label: str,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> None:
        if self.enable_profiling and self._profile_active:
            self._profile_events.append(_ProfileEventPair(label, start, end))

    def _record_preprocess_profile(
        self,
        events: PreprocessTimingEvents | None,
        stage_end: torch.cuda.Event | None,
    ) -> None:
        if events is None or stage_end is None:
            return
        self._profile_pair("voxel2_ms", events.start, events.after_voxel2)
        self._profile_pair(
            "outlier_filter_ms",
            events.after_voxel2,
            events.after_outlier_filter,
        )
        self._profile_pair(
            "final_selection_ms",
            events.after_outlier_filter,
            events.after_exact_count,
        )
        self._profile_pair(
            "stage_scale_ms",
            events.after_exact_count,
            stage_end,
        )

    def resolve_profile_window(
        self,
        *,
        synchronize: bool = False,
    ) -> dict[str, float]:
        """Resolve the last inference window after downstream CUDA work is done.

        Normal benchmark code already synchronizes on the recovery end event, so
        ``synchronize=False`` avoids an additional fence. Set it to true only for
        standalone debugging.
        """
        if not self.enable_profiling:
            return {}
        if not self._profile_events:
            return {}
        if synchronize:
            self._profile_events[-1].end.synchronize()
        totals: dict[str, float] = {}
        for pair in self._profile_events:
            totals[pair.label] = totals.get(pair.label, 0.0) + float(
                pair.start.elapsed_time(pair.end)
            )
        totals["preprocess_ms"] = (
            totals.get("voxel2_ms", 0.0)
            + totals.get("outlier_filter_ms", 0.0)
            + totals.get("final_selection_ms", 0.0)
            + totals.get("stage_scale_ms", 0.0)
        )
        totals["graph_ms"] = (
            totals.get("encode_ms", 0.0) + totals.get("decode_ms", 0.0)
        )
        totals["profiled_runner_ms"] = (
            totals["preprocess_ms"] + totals["graph_ms"]
        )
        self._last_profile = dict(totals)
        self._profile_active = False
        return totals

    # ------------------------------------------------------------------
    # Integrated world-space preprocessing
    # ------------------------------------------------------------------

    @property
    def second_base_voxel_size_m(self) -> float:
        return self.preprocessor.second_base_voxel_size_m

    @property
    def second_candidate_ratio(self) -> float:
        return self.preprocessor.second_candidate_ratio

    @property
    def auto_spatial_scale(self) -> bool:
        return self.preprocessor.auto_spatial_scale

    @property
    def target_model_volume(self) -> float:
        return self.preprocessor.target_model_volume

    @property
    def fixed_spatial_scale(self) -> float:
        return self.preprocessor.fixed_spatial_scale

    @property
    def volume_epsilon(self) -> float:
        return self.preprocessor.volume_epsilon

    @property
    def final_selection(self) -> str:
        return self.preprocessor.final_selection

    @property
    def spatial_scale(self) -> float:
        return self.preprocessor.spatial_scale

    @property
    def spatial_calibration(self) -> dict[str, object] | None:
        return self.preprocessor.spatial_calibration

    @property
    def second_calibration(self) -> dict[str, object] | None:
        return self.preprocessor.second_calibration

    @property
    def target_candidate_count(self) -> int:
        return self.preprocessor.target_candidate_count

    @property
    def second_voxel_size_m(self) -> float | None:
        return self.preprocessor.second_voxel_size_m

    def reset_spatial_scale(self) -> None:
        self.reset()
        self.preprocessor.reset_spatial_scale()

    def reset_preprocess_calibration(self) -> None:
        self.reset()
        self.preprocessor.reset_calibration()

    def calibrate_world(self, frame: torch.Tensor) -> dict[str, object]:
        return self.preprocessor.calibrate(frame)

    def stage_world(
        self,
        frame: torch.Tensor,
        point_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Automatic spatial scaling may be used without an explicit calibrate()
        # call (e.g. ScenePredictor). Only that first frame collects AABB/finite
        # diagnostics and therefore incurs a GPU->CPU synchronization.
        needs_scale_calibration = (
            self.preprocessor.needs_spatial_scale_calibration
        )
        prepared = self.preprocessor.prepare(
            frame,
            point_ids,
            collect_diagnostics=needs_scale_calibration,
        )
        if needs_scale_calibration:
            self.preprocessor.fit_spatial_scale(prepared.world_points)

        scale = self.spatial_scale
        slot = self._next_slot
        reference = self.next_input
        reference.copy_(prepared.world_points.unsqueeze(0), non_blocking=True)
        if scale != 1.0:
            reference.mul_(scale)

        # Shape checks are metadata-only and stay on the hot path. Expensive
        # finite checks are owned by calibration/debug mode in preprocessing.
        if reference.shape != (self.batch_size, self.num_points, 3):
            raise RuntimeError(
                "Static CUDA input shape changed unexpectedly: "
                f"{tuple(reference.shape)}."
            )

        stage_end = None
        if self.enable_profiling and self._profile_active:
            stage_end = torch.cuda.Event(enable_timing=True)
            stage_end.record()
        self._record_preprocess_profile(prepared.timing_events, stage_end)

        self._slot_selection_indices[slot] = prepared.selection_indices
        self._slot_point_ids[slot] = prepared.point_ids
        info = dict(prepared.info)
        info["spatial_scale"] = scale
        if "world_extent" in info:
            info["model_extent"] = tuple(
                float(v) * scale for v in info["world_extent"]
            )
            info["model_volume"] = float(info["world_volume"]) * scale**3
        self._slot_preprocess_info[slot] = info
        return prepared.world_points

    def push_world(
        self,
        frame: torch.Tensor,
        point_ids: torch.Tensor | None = None,
    ):
        self.stage_world(frame, point_ids=point_ids)
        return self.replay_next()

    # ------------------------------------------------------------------
    # Fixed-size model-space API
    # ------------------------------------------------------------------

    @property
    def next_input(self) -> torch.Tensor:
        return self.input_a if self._next_slot == 0 else self.input_b

    def reset(self) -> None:
        """Reset temporal state without recapturing graphs/calibration."""
        self.preprocessor.reset_temporal_history()
        self._next_slot = 0
        self._previous_slot = None
        self._last_source_slot = None
        self._last_target_slot = None
        self._current_output = None
        self._current_flow = None
        self._current_warped = None
        self._profile_active = False
        self._profile_events = []
        self._last_profile = None

    def _replay_encode(self, slot: int) -> None:
        if self.enable_profiling and self._profile_active:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = end = None

        if slot == 0:
            self.encode_graph_a.replay()
        else:
            self.encode_graph_b.replay()

        if end is not None and start is not None:
            end.record()
            self._profile_pair("encode_ms", start, end)

    def replay_next(self):
        """Encode ``next_input`` and, after the first frame, decode one pair."""
        current_slot = self._next_slot
        self._replay_encode(current_slot)

        if self._previous_slot is None:
            self._previous_slot = current_slot
            self._next_slot = 1 - current_slot
            return None

        source_slot = self._previous_slot
        target_slot = current_slot

        if self.enable_profiling and self._profile_active:
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            decode_start.record()
        else:
            decode_start = decode_end = None

        if source_slot == 0 and target_slot == 1:
            self.decode_graph_ab.replay()
            self._current_output = self.output_ab
            self._current_flow = self.flow_ab
            self._current_warped = self.warped_ab
        elif source_slot == 1 and target_slot == 0:
            self.decode_graph_ba.replay()
            self._current_output = self.output_ba
            self._current_flow = self.flow_ba
            self._current_warped = self.warped_ba
        else:
            raise RuntimeError(
                "Streaming graph slots did not alternate as expected."
            )

        if decode_end is not None and decode_start is not None:
            decode_end.record()
            self._profile_pair("decode_ms", decode_start, decode_end)

        self._last_source_slot = source_slot
        self._last_target_slot = target_slot
        self._previous_slot = target_slot
        self._next_slot = 1 - target_slot
        return self._current_output

    def push(self, frame: torch.Tensor):
        """Legacy fixed-size model-space copy + replay API."""
        reference = self.next_input
        if frame.device != reference.device:
            raise ValueError(
                f"frame must be on {reference.device}, got {frame.device}."
            )
        if frame.dtype != reference.dtype or frame.shape != reference.shape:
            raise ValueError(
                f"frame must have shape {tuple(reference.shape)} and dtype "
                f"{reference.dtype}; got {tuple(frame.shape)} and {frame.dtype}."
            )
        reference.copy_(frame, non_blocking=True)
        return self.replay_next()

    # ------------------------------------------------------------------
    # Model-space outputs
    # ------------------------------------------------------------------

    def flow(self) -> torch.Tensor:
        if self._current_flow is None:
            raise RuntimeError("At least two frames are required.")
        return self._current_flow

    def warped_points(self) -> torch.Tensor:
        if self._current_warped is None:
            raise RuntimeError("At least two frames are required.")
        return self._current_warped

    def velocity(self) -> torch.Tensor:
        # Velocity is derived lazily from flow, so the decode CUDA graph does
        # not spend a kernel computing a duplicate quantity every frame.
        return self.flow() / self.dt_s

    def output(self):
        if self._current_output is None:
            raise RuntimeError("At least two frames are required.")
        return self._current_output

    def source_points(self) -> torch.Tensor:
        if self._last_source_slot is None:
            raise RuntimeError("At least two frames are required.")
        return self.input_a if self._last_source_slot == 0 else self.input_b

    def target_points(self) -> torch.Tensor:
        if self._last_target_slot is None:
            raise RuntimeError("At least two frames are required.")
        return self.input_a if self._last_target_slot == 0 else self.input_b

    # ------------------------------------------------------------------
    # World-space outputs
    # ------------------------------------------------------------------

    def flow_world(self) -> torch.Tensor:
        return self.flow() / self.spatial_scale

    def source_points_world(self) -> torch.Tensor:
        return self.source_points() / self.spatial_scale

    def target_points_world(self) -> torch.Tensor:
        return self.target_points() / self.spatial_scale

    def warped_points_world(self) -> torch.Tensor:
        return self.source_points_world() + self.flow_world()

    def velocity_world(self) -> torch.Tensor:
        return self.flow_world() / self.dt_s

    # ------------------------------------------------------------------
    # Selection metadata
    # ------------------------------------------------------------------

    def source_selection_indices(self) -> torch.Tensor:
        if self._last_source_slot is None:
            raise RuntimeError("At least two frames are required.")
        result = self._slot_selection_indices[self._last_source_slot]
        if result is None:
            raise RuntimeError("Source frame was not staged with stage_world().")
        return result

    def target_selection_indices(self) -> torch.Tensor:
        if self._last_target_slot is None:
            raise RuntimeError("At least two frames are required.")
        result = self._slot_selection_indices[self._last_target_slot]
        if result is None:
            raise RuntimeError("Target frame was not staged with stage_world().")
        return result

    def source_point_ids(self) -> torch.Tensor:
        if self._last_source_slot is None:
            raise RuntimeError("At least two frames are required.")
        result = self._slot_point_ids[self._last_source_slot]
        if result is None:
            raise RuntimeError("Source frame was not staged with point_ids.")
        return result

    def target_point_ids(self) -> torch.Tensor:
        if self._last_target_slot is None:
            raise RuntimeError("At least two frames are required.")
        result = self._slot_point_ids[self._last_target_slot]
        if result is None:
            raise RuntimeError("Target frame was not staged with point_ids.")
        return result

    def source_preprocess_info(self) -> dict[str, object]:
        if self._last_source_slot is None:
            raise RuntimeError("At least two frames are required.")
        result = self._slot_preprocess_info[self._last_source_slot]
        if result is None:
            raise RuntimeError("Source frame was not staged with stage_world().")
        return dict(result)

    def target_preprocess_info(self) -> dict[str, object]:
        if self._last_target_slot is None:
            raise RuntimeError("At least two frames are required.")
        result = self._slot_preprocess_info[self._last_target_slot]
        if result is None:
            raise RuntimeError("Target frame was not staged with stage_world().")
        return dict(result)
