"""World-space adaptive voxel-2, exact-count selection, and scale calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from difflow3d.ops.pointnet2 import pointnet2_utils
from .voxel_outlier import (
    CudaVoxelComponentOutlierFilter,
    resolve_voxel_outlier_statistics,
)


@dataclass(frozen=True)
class PreprocessTimingEvents:
    """CUDA events recorded without synchronizing the hot path."""

    start: torch.cuda.Event
    after_voxel2: torch.cuda.Event
    after_outlier_filter: torch.cuda.Event
    after_exact_count: torch.cuda.Event


@dataclass(frozen=True)
class PreparedModelInput:
    world_points: torch.Tensor
    selection_indices: torch.Tensor
    point_ids: torch.Tensor | None
    input_keep_mask: torch.Tensor
    info: dict[str, object]
    timing_events: PreprocessTimingEvents | None = None
    # Exact-count input (voxel-2 output or the raw frame) kept so the frame can
    # be re-selected at another bucket size (both frames of a streaming pair
    # must share one point count). ``candidate_point_ids`` are per candidate.
    candidates: torch.Tensor | None = None
    candidate_indices: torch.Tensor | None = None
    candidate_point_ids: torch.Tensor | None = None
    candidate_info: dict[str, object] | None = None


class AdaptivePointPreprocessor:
    """Convert variable-size world point clouds to fixed-size model anchors.

    State machine:
      N1 < K       -> deterministic repeat to K; skip voxel-2/selection
      N1 == K      -> direct
      K < N1 <= rK -> selected backend to K; skip voxel-2
      N1 > rK      -> adaptive voxel-2, optional component filter, then
                      repeat/direct/select to K

    ``final_selection`` controls only the N>K exact-count reduction:
      * ``fps``: PointNet++ farthest-point sampling.
      * ``uniform``: deterministic evenly spaced selection from the voxel-key
        ordered candidate representatives. This is O(K) and avoids iterative FPS.

    The adaptive voxel-2 resolution and spatial scale are calibrated once and
    frozen until :meth:`reset_calibration` is explicitly called.

    Expensive GPU->CPU diagnostics (finite checks, AABB extent/volume) are only
    collected during one-shot calibration or when ``validate_finite`` is enabled.
    Normal streaming does not call ``.item()``, ``.cpu()`` or synchronize CUDA.
    """

    def __init__(
        self,
        *,
        num_points: int,
        second_base_voxel_size_m: float,
        second_candidate_ratio: float,
        auto_spatial_scale: bool,
        target_model_volume: float,
        fixed_spatial_scale: float = 1.0,
        volume_epsilon: float = 1.0e-12,
        final_selection: str = "fps",
        outlier_filter_enabled: bool = False,
        outlier_filter_min_component_size_ratio: float = 0.05,
        enable_timing: bool = False,
        validate_finite: bool = False,
        point_buckets: tuple[int, ...] | list[int] | None = None,
        sort_anchors_morton: bool = False,
    ) -> None:
        if num_points < 1024:
            raise ValueError("DifFlow3D requires num_points >= 1024.")
        buckets = sorted({int(value) for value in (point_buckets or (num_points,))})
        if buckets[0] < 1024 or buckets[-1] > num_points:
            raise ValueError(
                "point_buckets must lie in [1024, num_points]; "
                f"got {buckets} with num_points={num_points}."
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
        final_selection = str(final_selection).lower()
        if final_selection not in {"fps", "uniform"}:
            raise ValueError("final_selection must be 'fps' or 'uniform'.")

        self.num_points = int(num_points)
        self.second_base_voxel_size_m = float(second_base_voxel_size_m)
        self.second_candidate_ratio = float(second_candidate_ratio)
        self.target_candidate_count = int(
            math.ceil(self.second_candidate_ratio * self.num_points)
        )
        self.auto_spatial_scale = bool(auto_spatial_scale)
        self.target_model_volume = float(target_model_volume)
        self.fixed_spatial_scale = float(fixed_spatial_scale)
        self.volume_epsilon = float(volume_epsilon)
        self.final_selection = final_selection
        self.outlier_filter_enabled = bool(outlier_filter_enabled)
        if (
            self.outlier_filter_enabled
            and not pointnet2_utils.has_voxel_component_filter_op()
        ):
            raise RuntimeError(
                "Outlier filtering requires the updated PointNet2 extension; "
                "run: bash scripts/build_pointnet2_ops.sh"
            )
        self._outlier_filter = CudaVoxelComponentOutlierFilter(
            min_component_size_ratio=outlier_filter_min_component_size_ratio,
        )
        self.enable_timing = bool(enable_timing)
        self.validate_finite = bool(validate_finite)

        # Exact anchor counts the model may run at. A frame is reduced to the
        # largest bucket <= its candidate count (no duplication); only frames
        # below the smallest bucket are repeat-padded.
        self.point_buckets: tuple[int, ...] = tuple(buckets)
        # Sort the anchors along a Morton (Z-order) curve so that a strided
        # subset of them is spatially uniform (fast top-level selection).
        self.sort_anchors_morton = bool(sort_anchors_morton)
        # Exact-count selection indices are device-local and reused across frames.
        self._uniform_positions: dict[tuple[int, str], torch.Tensor] = {}

        self._second_auto_dimension = 2.0
        self._second_auto_iterations = 8
        self._second_auto_tolerance = 0.10
        self.second_voxel_size_m: float | None = None
        self._second_calibration: dict[str, object] | None = None
        self._spatial_scale: float | None = (
            None if self.auto_spatial_scale else self.fixed_spatial_scale
        )
        self._spatial_calibration: dict[str, object] | None = None

    @property
    def needs_spatial_scale_calibration(self) -> bool:
        return self.auto_spatial_scale and self._spatial_scale is None

    @property
    def spatial_scale(self) -> float:
        if self._spatial_scale is None:
            raise RuntimeError("Spatial scale has not been calibrated yet.")
        return float(self._spatial_scale)

    @property
    def spatial_calibration(self) -> dict[str, object] | None:
        return (
            None
            if self._spatial_calibration is None
            else dict(self._spatial_calibration)
        )

    @property
    def second_calibration(self) -> dict[str, object] | None:
        return (
            None
            if self._second_calibration is None
            else dict(self._second_calibration)
        )

    def reset_spatial_scale(self) -> None:
        self._spatial_scale = (
            None if self.auto_spatial_scale else self.fixed_spatial_scale
        )
        self._spatial_calibration = None

    def reset_calibration(self) -> None:
        self.second_voxel_size_m = None
        self._second_calibration = None
        self.reset_spatial_scale()

    @staticmethod
    def _validate_points(
        frame: torch.Tensor,
        *,
        check_finite: bool = False,
    ) -> torch.Tensor:
        if frame.ndim == 3:
            if frame.shape[0] != 1:
                raise ValueError("Expected [N,3] or [1,N,3].")
            frame = frame[0]
        if frame.ndim != 2 or frame.shape[1] != 3 or frame.shape[0] < 1:
            raise ValueError(
                f"Expected non-empty [N,3], got {tuple(frame.shape)}."
            )
        if frame.dtype != torch.float32:
            raise ValueError(f"World points must be float32, got {frame.dtype}.")
        if frame.device.type != "cuda":
            raise ValueError("World points must be CUDA tensors.")
        # This is deliberately optional: ``.item()`` introduces a device/host
        # synchronization. Calibration/debug paths may request it explicitly.
        if check_finite and not bool(torch.isfinite(frame).all().item()):
            raise ValueError("World points contain NaN or Inf.")
        return frame.contiguous()

    @staticmethod
    def _validate_point_ids(
        point_ids: torch.Tensor | None,
        count: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if point_ids is None:
            return None
        if point_ids.ndim == 2 and point_ids.shape[0] == 1:
            point_ids = point_ids[0]
        if point_ids.ndim != 1 or point_ids.shape[0] != count:
            raise ValueError(
                f"point_ids must have shape [{count}], got {tuple(point_ids.shape)}."
            )
        if point_ids.device != device:
            raise ValueError(
                f"point_ids must be on {device}, got {point_ids.device}."
            )
        return point_ids.contiguous()

    @staticmethod
    def _voxel_downsample(
        points: torch.Tensor,
        voxel_size: float,
        point_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        coordinates = torch.floor(points / float(voxel_size)).to(torch.int64)
        spatial_shifted = coordinates - coordinates.amin(dim=0)
        spatial_extents = spatial_shifted.amax(dim=0) + 1
        voxel_ids = (
            point_ids.to(torch.int64)
            if point_ids is not None
            else torch.full(
                (points.shape[0],),
                -1,
                device=points.device,
                dtype=torch.int64,
            )
        )
        # Dense rank of each point's track id (== torch.unique(return_inverse)),
        # computed with sort + cumsum + scatter so it needs no host sync.
        # torch.unique on CUDA copies the unique count back to the host, which
        # drained the whole stream once per stage_world.
        sorted_ids, id_order = torch.sort(voxel_ids)
        id_new = torch.ones_like(sorted_ids, dtype=torch.bool)
        id_new[1:] = sorted_ids[1:] != sorted_ids[:-1]
        ranks_sorted = torch.cumsum(id_new.to(torch.int64), dim=0) - 1
        track_ranks = torch.empty_like(voxel_ids)
        track_ranks.scatter_(0, id_order, ranks_sorted)

        # Separate tracks by two empty x slices. The unchanged 3-D CUDA
        # component operator can then label each instance independently while
        # still using one launch for the complete scene.
        component_coords = spatial_shifted.clone()
        track_stride = spatial_extents[0] + 2
        component_coords[:, 0] += track_ranks * track_stride
        component_extents = component_coords.amax(dim=0) + 1
        keys = (
            component_coords[:, 0]
            * (component_extents[1] * component_extents[2])
            + component_coords[:, 1] * component_extents[2]
            + component_coords[:, 2]
        )
        sorted_keys, order = torch.sort(keys)
        keep = torch.ones_like(sorted_keys, dtype=torch.bool)
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
        # One nonzero() (the single unavoidable host sync: the candidate count
        # drives the selection branch downstream) shared by both compactions,
        # instead of two independent boolean-mask gathers.
        keep_index = keep.nonzero(as_tuple=True)[0]
        indices = order.index_select(0, keep_index)
        sorted_group_ids = torch.cumsum(keep.to(torch.int64), dim=0) - 1
        input_to_candidate = torch.empty_like(order)
        input_to_candidate.scatter_(0, order, sorted_group_ids)
        unique_keys = sorted_keys.index_select(0, keep_index).contiguous()
        candidate_coords = (
            component_coords.index_select(0, indices).to(torch.int32).contiguous()
        )
        absolute_coords = (
            coordinates.index_select(0, indices).to(torch.int32).contiguous()
        )
        candidate_track_ranks = (
            track_ranks.index_select(0, indices).to(torch.int32).contiguous()
        )
        return (
            points.index_select(0, indices).contiguous(),
            indices.contiguous(),
            candidate_coords,
            absolute_coords,
            unique_keys,
            component_extents.to(torch.int32).contiguous(),
            input_to_candidate.contiguous(),
            candidate_track_ranks,
        )

    def _calibrate_second_voxel(
        self,
        points: torch.Tensor,
        point_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """One-shot untimed voxel-2 resolution search."""
        count = int(points.shape[0])
        report: dict[str, object] = {
            "input_count": count,
            "target_candidate_count": self.target_candidate_count,
            "mode": "auto-bypass",
            "trials": [],
        }
        if count <= self.target_candidate_count:
            report.update(
                candidate_count=count,
                second_voxel_resolution_m=None,
            )
            self._second_calibration = report
            return report

        base = self.second_base_voxel_size_m
        resolution = float(
            min(
                max(
                    base
                    * (count / self.target_candidate_count)
                    ** (1.0 / self._second_auto_dimension),
                    base * 1.001,
                ),
                base * 8.0,
            )
        )
        minimum, maximum = base * 1.001, base * 8.0
        best_resolution = best_count = None
        best_score = float("inf")
        fallback_resolution = fallback_count = None
        fallback_score = float("inf")
        trials: list[dict[str, float | int]] = []

        for _ in range(self._second_auto_iterations):
            candidate_points, _, _, _, _, _, _, _ = self._voxel_downsample(
                points, resolution, point_ids
            )
            candidate_count = int(candidate_points.shape[0])
            trials.append(
                {
                    "resolution_m": float(resolution),
                    "candidate_count": candidate_count,
                }
            )
            score = abs(
                math.log(max(candidate_count, 1) / self.target_candidate_count)
            )
            if score < fallback_score:
                fallback_score = float(score)
                fallback_resolution = float(resolution)
                fallback_count = candidate_count
            if candidate_count >= self.num_points and score < best_score:
                best_score = float(score)
                best_resolution = float(resolution)
                best_count = candidate_count
            relative_error = (
                abs(candidate_count - self.target_candidate_count)
                / self.target_candidate_count
            )
            if relative_error <= self._second_auto_tolerance:
                break
            correction = (
                max(candidate_count, 1) / self.target_candidate_count
            ) ** (1.0 / self._second_auto_dimension)
            correction = float(min(max(correction, 0.70), 1.50))
            new_resolution = float(
                min(max(resolution * correction, minimum), maximum)
            )
            if math.isclose(
                new_resolution,
                resolution,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            ):
                break
            resolution = new_resolution

        if best_resolution is None:
            if fallback_resolution is None or fallback_count is None:
                raise RuntimeError("Voxel-2 calibration produced no valid trial.")
            best_resolution, best_count = fallback_resolution, fallback_count

        self.second_voxel_size_m = float(best_resolution)
        report.update(
            mode="auto",
            second_voxel_resolution_m=self.second_voxel_size_m,
            candidate_count=int(best_count),
            trials=trials,
        )
        self._second_calibration = report
        return report

    def _second_downsample(
        self,
        points: torch.Tensor,
        point_ids: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        bool,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        count = int(points.shape[0])
        if count <= self.target_candidate_count:
            identity = torch.arange(
                count, device=points.device, dtype=torch.long
            )
            return points, identity, False, None, None, None, None, None, None

        if self.second_voxel_size_m is None:
            self._calibrate_second_voxel(points, point_ids)
        if self.second_voxel_size_m is None:
            identity = torch.arange(
                count, device=points.device, dtype=torch.long
            )
            return points, identity, False, None, None, None, None, None, None

        (
            candidates,
            indices,
            coords,
            absolute_coords,
            keys,
            extents,
            input_to_candidate,
            candidate_track_ranks,
        ) = (
            self._voxel_downsample(
                points,
                self.second_voxel_size_m,
                point_ids,
            )
        )
        return (
            candidates,
            indices,
            True,
            coords,
            absolute_coords,
            keys,
            extents,
            input_to_candidate,
            candidate_track_ranks,
        )

    def bucket_for(self, candidate_count: int) -> int:
        """Largest configured bucket <= ``candidate_count`` (smallest bucket below it)."""
        chosen = self.point_buckets[0]
        for bucket in self.point_buckets:
            if bucket <= candidate_count:
                chosen = bucket
        return int(chosen)

    def _repeat(
        self,
        points: torch.Tensor,
        target: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        count = int(points.shape[0])
        base = torch.arange(count, device=points.device, dtype=torch.long)
        indices = base.repeat(
            (target + count - 1) // count
        )[:target]
        return (
            points.index_select(0, indices).contiguous(),
            indices.contiguous(),
            count,
        )

    def _fps(
        self,
        points: torch.Tensor,
        target: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = pointnet2_utils.furthest_point_sample(
            points.unsqueeze(0).contiguous(),
            int(target),
        )[0].long()
        return (
            points.index_select(0, indices).contiguous(),
            indices.contiguous(),
        )

    def _uniform_select(
        self,
        points: torch.Tensor,
        target: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministically reduce N>K voxel representatives to exactly K."""
        count = int(points.shape[0])
        target = int(target)
        if count <= target:
            raise ValueError("_uniform_select requires more than target points.")

        key = (target, str(points.device))
        positions = self._uniform_positions.get(key)
        if positions is None:
            positions = torch.arange(
                target, device=points.device, dtype=torch.long
            )
            self._uniform_positions[key] = positions

        # Integer-rounded linspace [0, N-1]. Because N>K, indices stay unique
        # and both endpoints are retained.
        denominator = target - 1
        indices = torch.div(
            positions * (count - 1) + denominator // 2,
            denominator,
            rounding_mode="floor",
        )
        return (
            points.index_select(0, indices).contiguous(),
            indices.contiguous(),
        )

    def _select_exact_count(
        self,
        points: torch.Tensor,
        target: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.final_selection == "fps":
            return self._fps(points, target)
        return self._uniform_select(points, target)

    @staticmethod
    def _spread_bits(value: torch.Tensor) -> torch.Tensor:
        """Interleave the low 21 bits of an int64 tensor with two zero bits."""
        value = value & 0x1FFFFF
        value = (value | (value << 32)) & 0x1F00000000FFFF
        value = (value | (value << 16)) & 0x1F0000FF0000FF
        value = (value | (value << 8)) & 0x100F00F00F00F00F
        value = (value | (value << 4)) & 0x10C30C30C30C30C3
        value = (value | (value << 2)) & 0x1249249249249249
        return value

    def _morton_order(self, points: torch.Tensor) -> torch.Tensor:
        """Permutation sorting ``[N,3]`` points along a Z-order curve (no host sync)."""
        cell = self.second_voxel_size_m or self.second_base_voxel_size_m
        quantized = torch.floor(
            (points - points.min(dim=0).values) / float(cell)
        ).to(torch.int64).clamp_(0, (1 << 21) - 1)
        key = (
            self._spread_bits(quantized[:, 0])
            | (self._spread_bits(quantized[:, 1]) << 1)
            | (self._spread_bits(quantized[:, 2]) << 2)
        )
        return torch.argsort(key)

    def _reduce_to_count(
        self,
        candidates: torch.Tensor,
        target: int,
        mode_prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        """Repeat / pass through / select ``candidates`` to exactly ``target`` anchors."""
        count = int(candidates.shape[0])
        target = int(target)
        if count < target:
            anchors, local_indices, unique_count = self._repeat(candidates, target)
            selection_mode = f"{mode_prefix}-repeat"
        elif count == target:
            anchors = candidates.contiguous()
            local_indices = torch.arange(
                target,
                device=candidates.device,
                dtype=torch.long,
            )
            unique_count = target
            selection_mode = f"{mode_prefix}-direct"
        else:
            anchors, local_indices = self._select_exact_count(candidates, target)
            unique_count = target
            selection_mode = f"{mode_prefix}-{self.final_selection}"
        if self.sort_anchors_morton:
            order = self._morton_order(anchors)
            anchors = anchors.index_select(0, order).contiguous()
            local_indices = local_indices.index_select(0, order).contiguous()
        return anchors, local_indices, unique_count, selection_mode

    @staticmethod
    def aabb_extent_volume(
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        extent = points.amax(dim=0) - points.amin(dim=0)
        return extent, extent.prod()

    @staticmethod
    def _diagnostics(points: torch.Tensor) -> dict[str, object]:
        if not bool(torch.isfinite(points).all().item()):
            raise RuntimeError("Final DifFlow3D anchors contain NaN or Inf.")
        extent, volume = AdaptivePointPreprocessor.aabb_extent_volume(points)
        return {
            "world_extent": tuple(
                float(v) for v in extent.detach().cpu().tolist()
            ),
            "world_volume": float(volume.item()),
        }

    def fit_spatial_scale(self, anchors_world: torch.Tensor) -> None:
        diagnostics = self._diagnostics(anchors_world)
        volume = float(diagnostics["world_volume"])
        if not math.isfinite(volume) or volume <= self.volume_epsilon:
            raise RuntimeError(
                "Cannot auto-calibrate spatial scale from AABB volume "
                f"{volume:.9e}."
            )
        fitted = (self.target_model_volume / volume) ** (1.0 / 3.0)
        if not math.isfinite(fitted) or fitted <= 0.0:
            raise RuntimeError(f"Invalid automatic spatial scale: {fitted}.")
        self._spatial_scale = float(fitted)
        model_volume = volume * fitted**3
        self._spatial_calibration = {
            **diagnostics,
            "target_model_volume": self.target_model_volume,
            "spatial_scale": float(fitted),
            "model_volume": float(model_volume),
            "relative_volume_error": float(
                abs(model_volume - self.target_model_volume)
                / self.target_model_volume
            ),
        }

    def prepare(
        self,
        frame: torch.Tensor,
        point_ids: torch.Tensor | None = None,
        *,
        collect_diagnostics: bool = False,
        target_count: int | None = None,
    ) -> PreparedModelInput:
        """Reduce a world frame to exactly ``target_count`` anchors (default: its bucket)."""
        frame = self._validate_points(
            frame,
            check_finite=(collect_diagnostics or self.validate_finite),
        )
        point_ids = self._validate_point_ids(
            point_ids,
            int(frame.shape[0]),
            frame.device,
        )
        n1 = int(frame.shape[0])
        second_used = False
        filter_info: dict[str, object] = {
            "outlier_filter_enabled": self.outlier_filter_enabled,
            "outlier_filter_applied": False,
            "outlier_filter_input_count": n1,
            "outlier_filter_retained_count": n1,
            "outlier_filter_removed_count": 0,
        }
        candidate_count_before_filter = n1
        input_keep_mask = torch.ones(
            n1, device=frame.device, dtype=torch.bool
        )

        timing_events: PreprocessTimingEvents | None = None
        if self.enable_timing:
            event_start = torch.cuda.Event(enable_timing=True)
            event_after_voxel2 = torch.cuda.Event(enable_timing=True)
            event_after_outlier = torch.cuda.Event(enable_timing=True)
            event_after_exact = torch.cuda.Event(enable_timing=True)
            event_start.record()
        else:
            event_start = None
            event_after_voxel2 = None
            event_after_outlier = None
            event_after_exact = None

        if n1 <= self.target_candidate_count:
            # Small frames skip voxel-2: every input point is a candidate.
            candidates = frame
            candidate_indices = torch.arange(
                n1, device=frame.device, dtype=torch.long
            )
            mode_prefix = "first"
            if event_after_voxel2 is not None:
                event_after_voxel2.record()
            if event_after_outlier is not None:
                event_after_outlier.record()
        else:
            mode_prefix = "voxel2"
            (
                candidates,
                candidate_indices,
                second_used,
                candidate_coords,
                candidate_absolute_coords,
                candidate_keys,
                voxel_extents,
                input_to_candidate,
                candidate_track_ranks,
            ) = self._second_downsample(frame, point_ids)
            if event_after_voxel2 is not None:
                event_after_voxel2.record()
            candidate_count_before_filter = int(candidates.shape[0])
            filter_info.update(
                outlier_filter_input_count=candidate_count_before_filter,
                outlier_filter_retained_count=candidate_count_before_filter,
            )

            if self.outlier_filter_enabled and second_used:
                assert candidate_coords is not None
                assert candidate_absolute_coords is not None
                assert candidate_keys is not None
                assert voxel_extents is not None
                assert input_to_candidate is not None
                assert candidate_track_ranks is not None
                (
                    candidates,
                    retained_indices,
                    candidate_keep,
                    component_info,
                ) = (
                    self._outlier_filter.filter(
                        candidates,
                        candidate_coords,
                        candidate_absolute_coords,
                        candidate_keys,
                        voxel_extents,
                        candidate_track_ranks,
                    )
                )
                input_keep_mask = candidate_keep.index_select(
                    0, input_to_candidate
                ).contiguous()
                candidate_indices = candidate_indices.index_select(
                    0, retained_indices
                ).contiguous()
                filter_info.update(component_info)
                filter_info["outlier_filter_applied"] = True

            if event_after_outlier is not None:
                event_after_outlier.record()
        candidate_point_ids = (
            point_ids.index_select(0, candidate_indices).contiguous()
            if point_ids is not None
            else None
        )
        candidate_info: dict[str, object] = {
            "input_count": n1,
            "candidate_count_before_outlier_filter": (
                candidate_count_before_filter
            ),
            "candidate_count": int(candidates.shape[0]),
            "target_candidate_count": self.target_candidate_count,
            "second_voxel_used": bool(second_used),
            "second_mode": "auto" if second_used else "bypass",
            "second_voxel_resolution_m": (
                self.second_voxel_size_m if second_used else None
            ),
            "mode_prefix": mode_prefix,
            "final_selection": self.final_selection,
            "point_buckets": self.point_buckets,
            **filter_info,
        }
        target = (
            int(target_count)
            if target_count is not None
            else self.bucket_for(int(candidates.shape[0]))
        )
        anchors, local_indices, unique_count, selection_mode = self._reduce_to_count(
            candidates,
            target,
            mode_prefix,
        )

        if event_after_exact is not None:
            event_after_exact.record()
            assert event_start is not None
            assert event_after_voxel2 is not None
            assert event_after_outlier is not None
            timing_events = PreprocessTimingEvents(
                event_start,
                event_after_voxel2,
                event_after_outlier,
                event_after_exact,
            )

        if self.validate_finite and not collect_diagnostics:
            if not bool(torch.isfinite(anchors).all().item()):
                raise RuntimeError("Invalid final DifFlow3D model anchors.")

        return self._pack(
            anchors,
            local_indices,
            unique_count,
            selection_mode,
            candidates,
            candidate_indices,
            candidate_point_ids,
            candidate_info,
            input_keep_mask,
            timing_events,
            collect_diagnostics,
        )

    def _pack(
        self,
        anchors: torch.Tensor,
        local_indices: torch.Tensor,
        unique_count: int,
        selection_mode: str,
        candidates: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_point_ids: torch.Tensor | None,
        candidate_info: dict[str, object],
        input_keep_mask: torch.Tensor,
        timing_events: PreprocessTimingEvents | None,
        collect_diagnostics: bool,
    ) -> PreparedModelInput:
        target = int(anchors.shape[0])
        # Metadata-only checks; no CUDA synchronization.
        if anchors.shape != (target, 3) or target < 1024:
            raise RuntimeError(
                f"Preprocessing produced {tuple(anchors.shape)}; DifFlow3D "
                "model input must be [N>=1024, 3]."
            )
        final_indices = candidate_indices.index_select(
            0, local_indices
        ).contiguous()
        anchor_ids = (
            candidate_point_ids.index_select(0, local_indices).contiguous()
            if candidate_point_ids is not None
            else None
        )
        info: dict[str, object] = {
            **candidate_info,
            "anchor_count": target,
            "bucket": target,
            "unique_anchor_count": int(unique_count),
            "selection_mode": selection_mode,
        }
        info.pop("mode_prefix", None)
        if collect_diagnostics:
            info.update(self._diagnostics(anchors))
            resolve_voxel_outlier_statistics(info)
        return PreparedModelInput(
            anchors,
            final_indices,
            anchor_ids,
            input_keep_mask,
            info,
            timing_events,
            candidates,
            candidate_indices,
            candidate_point_ids,
            candidate_info,
        )

    def refinalize(
        self,
        prepared: PreparedModelInput,
        target_count: int,
    ) -> PreparedModelInput:
        """Re-select an already prepared frame at another exact anchor count."""
        if prepared.candidates is None or prepared.candidate_indices is None:
            raise ValueError("refinalize needs a PreparedModelInput with candidates.")
        candidate_info = dict(prepared.candidate_info or {})
        anchors, local_indices, unique_count, selection_mode = self._reduce_to_count(
            prepared.candidates,
            int(target_count),
            str(candidate_info.get("mode_prefix", "first")),
        )
        return self._pack(
            anchors,
            local_indices,
            unique_count,
            selection_mode,
            prepared.candidates,
            prepared.candidate_indices,
            prepared.candidate_point_ids,
            candidate_info,
            prepared.input_keep_mask,
            None,
            False,
        )

    def calibrate(self, frame: torch.Tensor) -> dict[str, object]:
        prepared = self.prepare(frame, collect_diagnostics=True)
        if self.auto_spatial_scale and self._spatial_scale is None:
            self.fit_spatial_scale(prepared.world_points)
        elif self._spatial_scale is not None and self._spatial_calibration is None:
            diagnostics = self._diagnostics(prepared.world_points)
            scale = float(self._spatial_scale)
            world_volume = float(diagnostics["world_volume"])
            self._spatial_calibration = {
                **diagnostics,
                "target_model_volume": self.target_model_volume,
                "spatial_scale": scale,
                "model_volume": world_volume * scale**3,
            }

        second = dict(self._second_calibration or {})
        if not second:
            second = {
                "input_count": int(prepared.info["input_count"]),
                "target_candidate_count": int(
                    prepared.info["target_candidate_count"]
                ),
                "mode": "bypass",
                "second_voxel_resolution_m": None,
                "candidate_count": int(prepared.info["candidate_count"]),
                "trials": [],
            }
        result = {
            "second": second,
            "spatial": self.spatial_calibration,
            "anchor_info": prepared.info,
        }
        return result
