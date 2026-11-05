"""CUDA-Graph streaming runner for production DifFlow3D inference."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from difflow3d.model import EncodedFrame, PointConvBidirection
from difflow3d.model.encoder import REFERENCE_POINTS, check_input_points
from difflow3d.model.segments import Segments, level1_capacities, level1_capacity
from .normalization import AnchoredNormalization, Transform
from .segmentation import MAX_SEGMENTS, SegmentNormalization, SegmentTransform, level1_total
from .preprocessing import (
    AdaptivePointPreprocessor,
    PreparedModelInput,
    PreprocessTimingEvents,
)


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


CAPACITY_HYSTERESIS = 1024     # level-1 padding a pair tolerates before it switches to a smaller multi-object graph


@dataclass
class _BucketGraphs:
    """Static inputs, captured CUDA graphs and outputs for one graph shape: (N0 anchors, N1 level-1 points).

    The anchors of a frame are one or several objects (segments) in object-major order. A multi-object graph
    has a fixed N0 and level-1 capacity N1 (model.segments.level1_capacity); the per-object layout of each slot's
    frame is a static `Segments` buffer set per slot (written by _stage_into), since the two frames of a pair
    split the bucket differently. The per-point canonical transform (center, scale) of each staged frame lives in
    static buffers too, so the outputs can be mapped back per object."""

    num_points: int
    level1_points: int
    input_a: torch.Tensor
    input_b: torch.Tensor
    encode_graph_a: torch.cuda.CUDAGraph
    encode_graph_b: torch.cuda.CUDAGraph
    decode_graph_ab: torch.cuda.CUDAGraph
    decode_graph_ba: torch.cuda.CUDAGraph
    segments_a: Segments | None = None
    segments_b: Segments | None = None
    center_a: torch.Tensor | None = None
    center_b: torch.Tensor | None = None
    scale_a: torch.Tensor | None = None
    scale_b: torch.Tensor | None = None
    encoded_a: EncodedFrame | None = None
    encoded_b: EncodedFrame | None = None
    output_ab: object = None
    output_ba: object = None
    flow_ab: torch.Tensor | None = None
    flow_ba: torch.Tensor | None = None
    warped_ab: torch.Tensor | None = None
    warped_ba: torch.Tensor | None = None
    # Cholesky factors [B,N,3,3] of the flow covariance (model units), only with an uncertainty head.
    chol_ab: torch.Tensor | None = None
    chol_ba: torch.Tensor | None = None

    def input(self, slot: int) -> torch.Tensor:
        return self.input_a if slot == 0 else self.input_b

    @property
    def multi_object(self) -> bool:
        return self.segments_a is not None

    def segments(self, slot: int) -> Segments | None:
        return self.segments_a if slot == 0 else self.segments_b

    def center(self, slot: int) -> torch.Tensor:
        return self.center_a if slot == 0 else self.center_b

    def scale(self, slot: int) -> torch.Tensor:
        return self.scale_a if slot == 0 else self.scale_b


class DifFlow3DStreamingCudaGraphRunner:
    """Double-buffered online CUDA Graph runner with deployment preprocessing.

    Hot-path invariants:
      * variable-size world clouds are reduced to exactly ``num_points``;
      * model coordinates use one frozen isotropic spatial scale, or the
        piecewise-constant canonical transform (``normalization``);
      * frame encodings are reused across adjacent pairs;
      * no GPU->CPU diagnostic synchronization occurs after calibration; with
        canonical normalisation, each staged frame copies 4 values (anchor
        centroid and RMS radius) to the host for the re-anchor decision;
      * optional detailed timing records CUDA events but does not synchronize;
      * the optional flow-uncertainty head runs inside the decode graphs; its
        scale input is a static device tensor refreshed by an on-device fill.
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
        outlier_filter_min_component_size_ratio: float = 0.05,
        enable_profiling: bool = False,
        validate_finite: bool = False,
        point_buckets: tuple[int, ...] | list[int] | None = None,
        multi_object_graphs: bool = True,
        sort_anchors_morton: bool = False,
        normalization: AnchoredNormalization | None = None,
        uncertainty_head: torch.nn.Module | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Streaming CUDA Graph inference requires CUDA.")
        if batch_size < 1 or warmup < 1:
            raise ValueError("The optimized streaming runner requires positive batch_size/warmup.")
        check_input_points(int(num_points))
        if second_base_voxel_size_m <= 0.0:
            raise ValueError("second_base_voxel_size_m must be positive.")
        if second_candidate_ratio <= 1.0:
            raise ValueError("second_candidate_ratio must be > 1.0.")
        if target_model_volume <= 0.0:
            raise ValueError("target_model_volume must be positive.")
        if fixed_spatial_scale <= 0.0:
            raise ValueError("fixed_spatial_scale must be positive.")
        if normalization is not None and auto_spatial_scale:
            raise ValueError("normalization replaces auto_spatial_scale; enable only one of them.")
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
            outlier_filter_min_component_size_ratio=(
                outlier_filter_min_component_size_ratio
            ),
            enable_timing=self.enable_profiling,
            validate_finite=validate_finite,
            point_buckets=point_buckets,
            sort_anchors_morton=sort_anchors_morton,
        )

        # One set of static inputs + graphs per graph shape (N0 anchors, N1 level-1 points, multi-object). Both
        # frames of a streaming pair run at the same shape: the smaller of their natural buckets; a frame already
        # encoded at another shape is re-selected from its candidates and re-encoded (one extra encode).
        # Single-object frames run the plain graphs (N1 = min(N0, 1024)); multi-object frames run segmented graphs
        # whose level 1 is quantised (segments.level1_capacities), all captured here so no frame pays a capture.
        self._warmup_count = int(warmup)
        self._buckets: dict[tuple[int, int, bool], _BucketGraphs] = {}
        for count in self.preprocessor.point_buckets:
            self._buckets[self._graph_key(int(count), (int(count),))] = self._new_graphs(int(count), min(int(count), REFERENCE_POINTS))
        self._largest_bucket = self._buckets[max(self._buckets)]
        multi_keys = ([(int(c), cap, True) for c in self.preprocessor.point_buckets for cap in level1_capacities(int(c))]
                      if multi_object_graphs else [])

        self._slot_selection_indices: list[torch.Tensor | None] = [None, None]
        self._slot_point_ids: list[torch.Tensor | None] = [None, None]
        self._slot_input_keep_masks: list[torch.Tensor | None] = [None, None]
        self._slot_preprocess_info: list[dict[str, object] | None] = [None, None]
        self._slot_prepared: list[PreparedModelInput | None] = [None, None]
        self._slot_bucket: list[int | None] = [None, None]
        self._slot_graphs: list[_BucketGraphs | None] = [None, None]
        # Canonical normalisation (optional): the similarity transform each slot was staged with.
        self.normalization = normalization
        # Per-object canonical normalisation (frames staged with point ids): same parameters as `normalization`.
        self.segment_normalization = (
            SegmentNormalization(normalization.model_center, normalization.model_radius, normalization.scale_ratio,
                                 normalization.center_shift, normalization.restage_previous)
            if normalization is not None else None
        )
        self._slot_transform: list[Transform | SegmentTransform | None] = [None, None]
        # Optional flow-uncertainty head, captured into each decode graph after the flow. Call contract
        # (difflow3d.runtime.uncertainty.FlowCovarianceHead), channels-last, model units:
        #   head(xyz [B,N,3], feat [B,N,128], u [B,N,1], flow [B,N,3], kk=[B]) -> L [B,N,3,3], Sigma_model = L L^T,
        # with xyz the source points, feat/u from RecurrentUnit.capture_head_inputs and kk the decoded pair's model
        # units per metre.
        self.uncertainty_head = uncertainty_head
        if uncertainty_head is not None:
            head_device = next(uncertainty_head.parameters()).device
            if head_device != device:
                raise ValueError(f"The uncertainty head is on {head_device}, the model on {device}.")
            uncertainty_head.eval()

        self._next_slot = 0
        self._previous_slot: int | None = None
        self._last_source_slot: int | None = None
        self._last_target_slot: int | None = None
        self._pair_bucket: int | None = None
        self._pair_graphs: _BucketGraphs | None = None   # graphs of the pair being staged (shape of both frames)
        self._pending_reencode_slot: int | None = None
        self._current_bucket: _BucketGraphs | None = None
        self._current_output = None
        self._current_flow: torch.Tensor | None = None
        self._current_warped: torch.Tensor | None = None
        self._current_chol: torch.Tensor | None = None
        # Model units per world unit the last pair was decoded at (None if push() decoded it before the automatic
        # spatial scale was calibrated).
        self._current_scale: float | None = None
        # Per-point scale [B,N] / centre [B,N,3] of the decoded pair's source frame (static buffers of its graph).
        self._current_scale_points: torch.Tensor | None = None
        self._current_center_points: torch.Tensor | None = None
        self.reencode_count = 0

        self._profile_active = False
        self._profile_events: list[_ProfileEventPair] = []

        # CUDA graph memory: one pool per graph role, shared by the graphs of every shape. Only one shape runs at
        # a time and a pair that switches shape re-encodes both slots in the new one (stage_world), so a graph
        # never reads another shape's outputs; within a shape the roles keep separate pools, because the a/b
        # encodings alternate and must not clobber each other. Captured largest first so smaller shapes reuse it.
        self._graph_pools = {role: torch.cuda.graph_pool_handle()
                             for role in ("encode_a", "encode_b", "decode_ab", "decode_ba")}
        self.capture_count = 0          # graph sets captured (a multi-object shape missing here is captured lazily)
        for count, capacity, _ in multi_keys:
            self._buckets[(count, capacity, True)] = self._new_multi_graphs(count, capacity)
        for key in sorted(self._buckets, key=lambda k: (k[0], k[1]), reverse=True):
            self._capture(self._buckets[key], int(warmup))
        self.reset()

    @staticmethod
    def _graph_key(count: int, segment_counts) -> tuple[int, int, bool]:
        """(N0, N1, multi-object): a single object runs the plain graph; several objects the segmented graph
        whose level-1 size is the quantised capacity of what the objects need."""
        if len(segment_counts) <= 1:
            return int(count), min(int(count), REFERENCE_POINTS), False
        return int(count), level1_capacity(int(count), level1_total([int(c) for c in segment_counts])), True

    def _new_graphs(self, count: int, level1: int) -> _BucketGraphs:
        shape = (self.batch_size, int(count), 3)
        input_a = torch.empty(shape, device=self.device, dtype=torch.float32)
        return _BucketGraphs(
            int(count), int(level1), input_a, torch.empty_like(input_a),
            torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(),
            center_a=torch.zeros(shape, device=self.device), center_b=torch.zeros(shape, device=self.device),
            scale_a=torch.ones((self.batch_size, int(count)), device=self.device),
            scale_b=torch.ones((self.batch_size, int(count)), device=self.device),
        )

    def _new_multi_graphs(self, count: int, capacity: int) -> _BucketGraphs:
        bucket = self._new_graphs(count, capacity)
        # A representative layout for capture (one object + padding): the segment tables are rewritten per frame.
        for name in ("segments_a", "segments_b"):
            setattr(bucket, name, Segments((int(count),), self.model.level1_points, self.device,
                                           level1_total=int(capacity)))
        return bucket

    def _graphs_for(self, count: int, segment_counts, capacity: int | None = None) -> _BucketGraphs:
        """Graphs of this shape (``capacity`` forces a multi-object shape), captured on first use."""
        key = (int(count), int(capacity), True) if capacity is not None else self._graph_key(count, segment_counts)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = (self._new_multi_graphs(key[0], key[1]) if key[2] else
                      self._new_graphs(key[0], key[1]))
            self._capture(bucket, self._warmup_count)
            self._buckets[key] = bucket
        return bucket

    # Legacy single-size accessors (largest bucket).
    @property
    def input_a(self) -> torch.Tensor:
        return self._largest_bucket.input_a

    @property
    def input_b(self) -> torch.Tensor:
        return self._largest_bucket.input_b

    @property
    def point_buckets(self) -> tuple[int, ...]:
        return tuple(sorted({key[0] for key in self._buckets}))

    @property
    def has_uncertainty(self) -> bool:
        return self.uncertainty_head is not None

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    @contextmanager
    def _head_input_capture(self):
        """Let the fine recurrent unit keep the head inputs while the head is warmed up or captured."""
        if self.uncertainty_head is None:
            yield
            return
        unit = self.model.recurrent0
        previous = unit.capture_head_inputs
        unit.capture_head_inputs = True
        try:
            yield
        finally:
            unit.capture_head_inputs = previous
            unit.head_inputs = None

    def _flow_cholesky(self, source: torch.Tensor, flow: torch.Tensor, scale: torch.Tensor, segments) -> torch.Tensor:
        """Uncertainty head on the inputs kept by the decode that just ran (channels-last, model units); kk is the
        per-point model-units-per-metre of the source frame (per-object normalisation)."""
        feat, certainty = self.model.recurrent0.head_inputs
        return self.uncertainty_head(
            source,
            feat.transpose(1, 2),
            certainty.transpose(1, 2),
            flow,
            kk=scale,
            seg=segments.ids[0] if segments is not None else None,
        )

    def _warmup(self, bucket: _BucketGraphs, count: int) -> None:
        current = torch.cuda.current_stream(self.device)
        setup = torch.cuda.Stream(device=self.device)
        setup.wait_stream(current)

        with torch.cuda.stream(setup), torch.inference_mode(), self._head_input_capture():
            for _ in range(count):
                encoded_a = self.model.encode_frame(bucket.input_a, bucket.input_a, segments=bucket.segments_a)
                encoded_b = self.model.encode_frame(bucket.input_b, bucket.input_b, segments=bucket.segments_b)
                output_ab = self.model.decode_pair(
                    encoded_a, encoded_b, None, self.uncertainty
                )
                flow_ab = output_ab[0][0][0].permute(0, 2, 1).contiguous()
                if self.uncertainty_head is not None:
                    self._flow_cholesky(bucket.input_a, flow_ab, bucket.scale_a, bucket.segments_a)
                output_ba = self.model.decode_pair(
                    encoded_b, encoded_a, None, self.uncertainty
                )
                flow_ba = output_ba[0][0][0].permute(0, 2, 1).contiguous()
                if self.uncertainty_head is not None:
                    self._flow_cholesky(bucket.input_b, flow_ba, bucket.scale_b, bucket.segments_b)

        current.wait_stream(setup)
        torch.cuda.synchronize(self.device)

    def _capture(self, bucket: _BucketGraphs, warmup: int) -> None:
        # RNG initialization deliberately occurs outside CUDA graph capture.
        bucket.input_a.normal_(0.0, 0.25)
        bucket.input_b.normal_(0.0, 0.25)
        self._warmup(bucket, warmup)

        with torch.cuda.graph(bucket.encode_graph_a, pool=self._graph_pools["encode_a"]):
            with torch.inference_mode():
                bucket.encoded_a = self.model.encode_frame(
                    bucket.input_a,
                    bucket.input_a,
                    segments=bucket.segments_a,
                )

        with torch.cuda.graph(bucket.encode_graph_b, pool=self._graph_pools["encode_b"]):
            with torch.inference_mode():
                bucket.encoded_b = self.model.encode_frame(
                    bucket.input_b,
                    bucket.input_b,
                    segments=bucket.segments_b,
                )

        bucket.encode_graph_a.replay()
        bucket.encode_graph_b.replay()
        torch.cuda.synchronize(self.device)

        assert bucket.encoded_a is not None
        assert bucket.encoded_b is not None

        with torch.cuda.graph(bucket.decode_graph_ab, pool=self._graph_pools["decode_ab"]):
            with torch.inference_mode(), self._head_input_capture():
                bucket.output_ab = self.model.decode_pair(
                    bucket.encoded_a,
                    bucket.encoded_b,
                    None,
                    self.uncertainty,
                )
                bucket.flow_ab = (
                    bucket.output_ab[0][0][0]
                    .permute(0, 2, 1)
                    .float()
                    .contiguous()
                )
                bucket.warped_ab = bucket.input_a + bucket.flow_ab
                if self.uncertainty_head is not None:
                    bucket.chol_ab = self._flow_cholesky(bucket.input_a, bucket.flow_ab, bucket.scale_a, bucket.segments_a)

        with torch.cuda.graph(bucket.decode_graph_ba, pool=self._graph_pools["decode_ba"]):
            with torch.inference_mode(), self._head_input_capture():
                bucket.output_ba = self.model.decode_pair(
                    bucket.encoded_b,
                    bucket.encoded_a,
                    None,
                    self.uncertainty,
                )
                bucket.flow_ba = (
                    bucket.output_ba[0][0][0]
                    .permute(0, 2, 1)
                    .float()
                    .contiguous()
                )
                bucket.warped_ba = bucket.input_b + bucket.flow_ba
                if self.uncertainty_head is not None:
                    bucket.chol_ba = self._flow_cholesky(bucket.input_b, bucket.flow_ba, bucket.scale_b, bucket.segments_b)

        torch.cuda.synchronize(self.device)
        # Streaming state is untouched: a lazy capture runs between staging a frame and replaying it.
        self.capture_count += 1

    # ------------------------------------------------------------------
    # Non-synchronizing detailed profiler
    # ------------------------------------------------------------------

    def begin_profile_window(self) -> None:
        if not self.enable_profiling:
            return
        self._profile_events = []
        self._profile_active = True

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

        slot = self._next_slot
        previous = self._previous_slot
        bucket = int(prepared.info["bucket"])
        if previous is not None:
            previous_prepared = self._slot_prepared[previous]
            assert previous_prepared is not None
            # The pair runs at the smaller natural bucket of its two frames.
            previous_natural = self.preprocessor.bucket_for(
                int(previous_prepared.candidates.shape[0])
            )
            # ... raised so that both frames' objects fit (segmentation.min_segment_total).
            segments = max(len(previous_prepared.segment_counts or (0,)), len(prepared.segment_counts or (0,)))
            bucket = self.preprocessor.fit_segments(min(bucket, previous_natural), segments)
            if int(prepared.info["bucket"]) != bucket:
                prepared = self.preprocessor.refinalize(prepared, bucket)
        transform, changed = self._frame_transform(prepared)
        previous_prepared = None
        if previous is not None:
            previous_prepared = self._slot_prepared[previous]
            if self._slot_bucket[previous] != bucket:
                previous_prepared = self.preprocessor.refinalize(previous_prepared, bucket)
        graphs = self._pair_graphs_for(bucket, previous_prepared, prepared)
        if previous is not None:
            # Both frames of a pair run one graph and share one transform: a frame encoded in another graph (bucket
            # or level-1 capacity), at another bucket, or before a re-anchor is re-staged (one extra encode).
            renormalize = (self.normalization is not None and self.normalization.restage_previous and changed)
            if (graphs is not self._slot_graphs[previous] or previous_prepared is not self._slot_prepared[previous]
                    or renormalize):
                self._stage_into(previous, previous_prepared, graphs, record_profile=False, transform=transform)
                self._pending_reencode_slot = previous
                self.reencode_count += 1
        self._pair_bucket = bucket
        self._pair_graphs = graphs
        self._stage_into(slot, prepared, graphs, record_profile=True, transform=transform)
        return prepared.world_points

    def _frame_transform(self, prepared: PreparedModelInput):
        """This frame's model transform and whether it changed: per object (anchored canonical, frames with
        point ids), the anchored canonical one for a single cloud, or the plain spatial scale."""
        world_points = prepared.world_points
        if self.normalization is not None and prepared.segment_counts is not None:
            return self.segment_normalization.update(prepared.segment_summary, world_points.device, world_points.dtype)
        if self.normalization is not None:
            return self.normalization.update(world_points)
        zero = torch.zeros(3, device=world_points.device, dtype=world_points.dtype)
        return Transform(center=zero, scale=float(self.spatial_scale), model_center=zero), False

    def _pair_graphs_for(self, count: int, previous: PreparedModelInput | None,
                         current: PreparedModelInput) -> _BucketGraphs:
        """The graph both frames of a pair run: plain for one object, otherwise the segmented graph whose level-1
        capacity fits both frames' object layouts (the pair has one object set: CommonInstanceFilter)."""
        layouts = [p.segment_counts or (count,) for p in (previous, current) if p is not None]
        if max(len(layout) for layout in layouts) <= 1:
            return self._graphs_for(count, (count,))
        need = max(level1_total(list(layout)) for layout in layouts)
        # Hysteresis: keep the previous frame's graph while it fits and pads less than CAPACITY_HYSTERESIS
        # (switching re-encodes the previous frame); the per-frame split moves the need across steps often.
        held = self._slot_graphs[self._previous_slot] if self._previous_slot is not None else None
        if (held is not None and held.multi_object and held.num_points == count
                and need <= held.level1_points < need + CAPACITY_HYSTERESIS):
            return held
        return self._graphs_for(count, layouts[-1], level1_capacity(count, need))

    def _stage_into(
        self,
        slot: int,
        prepared: PreparedModelInput,
        bucket: _BucketGraphs,
        *,
        record_profile: bool,
        transform: Transform,
    ) -> None:
        bucket_count = bucket.num_points
        if bucket.multi_object:
            # This slot's object layout (same N0, objects fitting the graph's level-1 capacity).
            bucket.segments(slot).write(tuple(prepared.segment_counts))
        reference = bucket.input(slot)
        reference.copy_(prepared.world_points.unsqueeze(0), non_blocking=True)
        center, scale = bucket.center(slot), bucket.scale(slot)
        if isinstance(transform, SegmentTransform):
            point_center, point_scale = transform.expand(prepared.segment_index)
            center.copy_(point_center.unsqueeze(0), non_blocking=True)
            scale.copy_(point_scale.unsqueeze(0), non_blocking=True)
            reference.sub_(center).mul_(scale.unsqueeze(2)).add_(transform.model_center)
        else:
            center.copy_(transform.center.reshape(1, 1, 3).expand_as(center))
            scale.fill_(float(transform.scale))
            if self.normalization is not None:
                reference.sub_(transform.center).mul_(transform.scale).add_(transform.model_center)
            elif float(transform.scale) != 1.0:
                reference.mul_(float(transform.scale))

        # Shape checks are metadata-only and stay on the hot path. Expensive
        # finite checks are owned by calibration/debug mode in preprocessing.
        if reference.shape != (self.batch_size, bucket_count, 3):
            raise RuntimeError(
                "Static CUDA input shape changed unexpectedly: "
                f"{tuple(reference.shape)}."
            )

        if record_profile:
            stage_end = None
            if self.enable_profiling and self._profile_active:
                stage_end = torch.cuda.Event(enable_timing=True)
                stage_end.record()
            self._record_preprocess_profile(prepared.timing_events, stage_end)

        self._slot_prepared[slot] = prepared
        self._slot_bucket[slot] = bucket_count
        self._slot_graphs[slot] = bucket
        self._slot_transform[slot] = transform
        self._slot_selection_indices[slot] = prepared.selection_indices
        self._slot_point_ids[slot] = prepared.point_ids
        self._slot_input_keep_masks[slot] = prepared.input_keep_mask
        info = dict(prepared.info)
        info["spatial_scale"] = float(transform.scale) if not isinstance(transform, SegmentTransform) else None
        info["segment_scales"] = ([self.normalization.model_radius / r for _, r in transform.per_object]
                                  if isinstance(transform, SegmentTransform) else None)
        if "world_extent" in info:
            info["model_extent"] = tuple(
                float(v) * scale for v in info["world_extent"]
            )
            info["model_volume"] = float(info["world_volume"]) * scale**3
        self._slot_preprocess_info[slot] = info
        return bucket

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
        """Reset streaming state without recapturing graphs/calibration."""
        self._next_slot = 0
        self._previous_slot = None
        self._last_source_slot = None
        self._last_target_slot = None
        self._pair_bucket = None
        self._pair_graphs = None
        self._pending_reencode_slot = None
        self._current_bucket = None
        self._slot_prepared = [None, None]
        self._slot_bucket = [None, None]
        self._slot_graphs = [None, None]
        self._slot_transform = [None, None]
        if self.normalization is not None:
            self.normalization.reset()
            self.segment_normalization.reset()
        self._current_output = None
        self._current_flow = None
        self._current_warped = None
        self._current_chol = None
        self._current_scale = None
        self._profile_active = False
        self._profile_events = []

    def _replay_encode(self, bucket: _BucketGraphs, slot: int) -> None:
        if self.enable_profiling and self._profile_active:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = end = None

        if slot == 0:
            bucket.encode_graph_a.replay()
        else:
            bucket.encode_graph_b.replay()

        if end is not None and start is not None:
            end.record()
            self._profile_pair("encode_ms", start, end)

    def replay_next(self):
        """Encode the staged frame and, after the first frame, decode one pair."""
        current_slot = self._next_slot
        if self._pair_bucket is None:
            raise RuntimeError("stage_world()/push() must run before replay_next().")
        bucket = self._pair_graphs
        if self._pending_reencode_slot is not None:
            # The previous frame was re-selected at this pair's bucket.
            self._replay_encode(bucket, self._pending_reencode_slot)
            self._pending_reencode_slot = None
        self._replay_encode(bucket, current_slot)

        if self._previous_slot is None:
            self._previous_slot = current_slot
            self._next_slot = 1 - current_slot
            return None

        source_slot = self._previous_slot
        target_slot = current_slot
        self._current_bucket = bucket
        scale = self._decode_scale(source_slot)

        if self.enable_profiling and self._profile_active:
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            decode_start.record()
        else:
            decode_start = decode_end = None

        if source_slot == 0 and target_slot == 1:
            bucket.decode_graph_ab.replay()
            self._current_output = bucket.output_ab
            self._current_flow = bucket.flow_ab
            self._current_warped = bucket.warped_ab
            self._current_chol = bucket.chol_ab
        elif source_slot == 1 and target_slot == 0:
            bucket.decode_graph_ba.replay()
            self._current_output = bucket.output_ba
            self._current_flow = bucket.flow_ba
            self._current_warped = bucket.warped_ba
            self._current_chol = bucket.chol_ba
        else:
            raise RuntimeError(
                "Streaming graph slots did not alternate as expected."
            )

        if decode_end is not None and decode_start is not None:
            decode_end.record()
            self._profile_pair("decode_ms", decode_start, decode_end)

        self._current_scale = scale
        # Per-object frames: snapshot of the source frame's per-point transform at decode time (the next
        # stage_world() may re-anchor, i.e. rewrite, the slot's live buffers, and the world outputs must keep the
        # transform this pair was decoded at). Single-cloud frames keep the scalar path (bit-identical outputs).
        per_object = isinstance(self._slot_transform[source_slot], SegmentTransform)
        self._current_scale_points = bucket.scale(source_slot).clone() if per_object else None
        self._current_center_points = bucket.center(source_slot).clone() if per_object else None
        self._last_source_slot = source_slot
        self._last_target_slot = target_slot
        self._previous_slot = target_slot
        self._next_slot = 1 - target_slot
        return self._current_output

    def _decode_scale(self, source_slot: int) -> float | None:
        """Model units per world unit of the pair about to be decoded (None while the automatic spatial scale is
        uncalibrated, which only the model-space push() API allows)."""
        if self.normalization is None and self.preprocessor.needs_spatial_scale_calibration:
            return None
        return self._slot_scale(source_slot)

    def push(self, frame: torch.Tensor):
        """Legacy fixed-size model-space copy + replay API (one bucket per stream)."""
        if frame.dim() != 3 or int(frame.shape[1]) not in self.point_buckets:
            raise ValueError(
                f"frame must have shape (batch, N, 3) with N in {self.point_buckets}; "
                f"got {tuple(frame.shape)}."
            )
        count = int(frame.shape[1])
        if self._pair_bucket is not None and self._pair_bucket != count:
            raise ValueError(
                "push() keeps one point count per stream; call reset() to change it."
            )
        graphs = self._buckets[self._graph_key(count, (count,))]
        self._pair_graphs = graphs
        reference = graphs.input(self._next_slot)
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
        # Model-space input: no centring; the (possibly uncalibrated) spatial scale maps the outputs back.
        graphs.center(self._next_slot).zero_()
        graphs.scale(self._next_slot).fill_(float(self.spatial_scale) if not self.preprocessor.needs_spatial_scale_calibration else 1.0)
        self._pair_bucket = count
        self._slot_bucket[self._next_slot] = count
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
        if self._last_source_slot is None or self._current_bucket is None:
            raise RuntimeError("At least two frames are required.")
        return self._current_bucket.input(self._last_source_slot)

    def target_points(self) -> torch.Tensor:
        if self._last_target_slot is None or self._current_bucket is None:
            raise RuntimeError("At least two frames are required.")
        return self._current_bucket.input(self._last_target_slot)

    def current_point_count(self) -> int:
        """Exact anchor count of the last decoded pair."""
        if self._current_bucket is None:
            raise RuntimeError("At least two frames are required.")
        return int(self._current_bucket.num_points)

    # ------------------------------------------------------------------
    # World-space outputs
    # ------------------------------------------------------------------

    def _to_world(self, points: torch.Tensor, slot: int | None) -> torch.Tensor:
        transform = self._slot_transform[slot] if slot is not None else None
        if self.normalization is None or transform is None:
            return points / self.spatial_scale
        if not isinstance(transform, SegmentTransform):
            return (points - transform.model_center) / transform.scale + transform.center
        if slot == self._last_source_slot and self._current_scale_points is not None:
            scale, center = self._current_scale_points, self._current_center_points     # decode-time snapshot
        else:
            bucket = self._current_bucket
            scale, center = bucket.scale(slot), bucket.center(slot)
        return (points - transform.model_center) / scale.unsqueeze(2) + center

    def _slot_scale(self, slot: int | None) -> float | None:
        """Model units per world unit of a staged single-cloud frame (None for per-object transforms: use the
        per-point scale buffers)."""
        transform = self._slot_transform[slot] if slot is not None else None
        if isinstance(transform, SegmentTransform):
            return None
        if self.normalization is not None and transform is not None:
            return transform.scale
        return self.spatial_scale

    def flow_world(self) -> torch.Tensor:
        """Flow of the last decoded pair in world units, with the scale that pair was decoded at: still valid after
        the next stage_world() (which may re-anchor the source slot), until the next replay_next()."""
        flow = self.flow()
        if self._current_scale_points is not None:
            return flow / self._current_scale_points.unsqueeze(2)
        scale = self._current_scale
        if scale is None:
            # push() before the automatic spatial scale was calibrated: the current scale (raises while uncalibrated).
            scale = self._slot_scale(self._last_source_slot)
        return flow / scale

    def source_points_world(self) -> torch.Tensor:
        return self._to_world(self.source_points(), self._last_source_slot)

    def target_points_world(self) -> torch.Tensor:
        return self._to_world(self.target_points(), self._last_target_slot)

    def warped_points_world(self) -> torch.Tensor:
        return self.source_points_world() + self.flow_world()

    def velocity_world(self) -> torch.Tensor:
        return self.flow_world() / self.dt_s

    # ------------------------------------------------------------------
    # Flow uncertainty (requires uncertainty_head)
    # ------------------------------------------------------------------

    def flow_chol_model(self) -> torch.Tensor:
        """Static ``[B,N,3,3]`` lower-triangular factor of the flow covariance in model units (overwritten by the next
        decode); world covariance = L L^T / k^2 with k the scale the pair was decoded at (the head's kk)."""
        if self.uncertainty_head is None:
            raise RuntimeError("The runner was built without an uncertainty head.")
        if self._current_chol is None:
            raise RuntimeError("At least two frames are required.")
        return self._current_chol

    def flow_cov_world(self) -> torch.Tensor:
        """New ``[B,N,3,3]`` covariance (m^2) of the displacement of each source anchor, row-aligned with flow_world();
        like flow_world() it uses the decode-time scale, so it stays valid until the next replay_next()."""
        chol = self.flow_chol_model()
        # Elementwise L L^T: exact fp32 and exactly symmetric (a batched GEMM may run in TF32).
        covariance = (chol.unsqueeze(-2) * chol.unsqueeze(-3)).sum(dim=-1)
        if self._current_scale_points is not None:
            return covariance / (self._current_scale_points ** 2)[..., None, None]
        scale = self._current_scale
        return covariance / (scale * scale)

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

    def source_input_keep_mask(self) -> torch.Tensor:
        """Mask over the source frame passed to :meth:`stage_world`."""
        if self._last_source_slot is None:
            raise RuntimeError("At least two frames are required.")
        result = self._slot_input_keep_masks[self._last_source_slot]
        if result is None:
            raise RuntimeError("Source frame was not staged with stage_world().")
        return result

    def target_input_keep_mask(self) -> torch.Tensor:
        """Mask over the target frame passed to :meth:`stage_world`."""
        if self._last_target_slot is None:
            raise RuntimeError("At least two frames are required.")
        result = self._slot_input_keep_masks[self._last_target_slot]
        if result is None:
            raise RuntimeError("Target frame was not staged with stage_world().")
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
