"""World-space adaptive voxel-2, exact-count selection, and scale calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from difflow3d.ops.pointnet2 import pointnet2_utils


@dataclass(frozen=True)
class PreprocessTimingEvents:
    """CUDA events recorded without synchronizing the hot path."""

    start: torch.cuda.Event
    after_voxel2: torch.cuda.Event
    after_exact_count: torch.cuda.Event


@dataclass(frozen=True)
class PreparedModelInput:
    world_points: torch.Tensor
    selection_indices: torch.Tensor
    point_ids: torch.Tensor | None
    info: dict[str, object]
    timing_events: PreprocessTimingEvents | None = None


class AdaptivePointPreprocessor:
    """Convert variable-size world point clouds to fixed-size model anchors.

    State machine:
      N1 < K       -> deterministic repeat to K; skip voxel-2/selection
      N1 == K      -> direct
      K < N1 <= rK -> selected backend to K; skip voxel-2
      N1 > rK      -> adaptive voxel-2, then repeat/direct/select to K

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
        enable_timing: bool = False,
        validate_finite: bool = False,
    ) -> None:
        if num_points < 1024:
            raise ValueError("DifFlow3D requires num_points >= 1024.")
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
        self.enable_timing = bool(enable_timing)
        self.validate_finite = bool(validate_finite)

        # Fixed-K selection indices are device-local and reused across frames.
        self._uniform_positions: torch.Tensor | None = None
        self._uniform_denominator = self.num_points - 1

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = torch.floor(points / float(voxel_size)).to(torch.int64)
        shifted = coordinates - coordinates.amin(dim=0)
        extents = shifted.amax(dim=0) + 1
        keys = (
            shifted[:, 0] * (extents[1] * extents[2])
            + shifted[:, 1] * extents[2]
            + shifted[:, 2]
        )
        sorted_keys, order = torch.sort(keys)
        keep = torch.ones_like(sorted_keys, dtype=torch.bool)
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
        indices = order[keep]
        return (
            points.index_select(0, indices).contiguous(),
            indices.contiguous(),
        )

    def _calibrate_second_voxel(
        self,
        points: torch.Tensor,
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
            candidate_points, _ = self._voxel_downsample(points, resolution)
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
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        count = int(points.shape[0])
        if count <= self.target_candidate_count:
            identity = torch.arange(
                count, device=points.device, dtype=torch.long
            )
            return points, identity, False

        if self.second_voxel_size_m is None:
            self._calibrate_second_voxel(points)
        if self.second_voxel_size_m is None:
            identity = torch.arange(
                count, device=points.device, dtype=torch.long
            )
            return points, identity, False

        candidates, indices = self._voxel_downsample(
            points, self.second_voxel_size_m
        )
        return candidates, indices, True

    def _repeat(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        count = int(points.shape[0])
        base = torch.arange(count, device=points.device, dtype=torch.long)
        indices = base.repeat(
            (self.num_points + count - 1) // count
        )[: self.num_points]
        return (
            points.index_select(0, indices).contiguous(),
            indices.contiguous(),
            count,
        )

    def _fps(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = pointnet2_utils.furthest_point_sample(
            points.unsqueeze(0).contiguous(),
            self.num_points,
        )[0].long()
        return (
            points.index_select(0, indices).contiguous(),
            indices.contiguous(),
        )

    def _uniform_select(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministically reduce N>K voxel representatives to exactly K."""
        count = int(points.shape[0])
        if count <= self.num_points:
            raise ValueError("_uniform_select requires more than num_points.")

        positions = self._uniform_positions
        if positions is None or positions.device != points.device:
            positions = torch.arange(
                self.num_points, device=points.device, dtype=torch.long
            )
            self._uniform_positions = positions

        # Integer-rounded linspace [0, N-1]. Because N>K, indices stay unique
        # and both endpoints are retained.
        denominator = self._uniform_denominator
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.final_selection == "fps":
            return self._fps(points)
        return self._uniform_select(points)

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
    ) -> PreparedModelInput:
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

        timing_events: PreprocessTimingEvents | None = None
        if self.enable_timing:
            event_start = torch.cuda.Event(enable_timing=True)
            event_after_voxel2 = torch.cuda.Event(enable_timing=True)
            event_after_exact = torch.cuda.Event(enable_timing=True)
            event_start.record()
        else:
            event_start = event_after_voxel2 = event_after_exact = None

        if n1 < self.num_points:
            candidates = frame
            candidate_indices = torch.arange(
                n1, device=frame.device, dtype=torch.long
            )
            anchors, local_indices, unique_count = self._repeat(candidates)
            selection_mode = "first-repeat"
            if event_after_voxel2 is not None:
                event_after_voxel2.record()
        elif n1 == self.num_points:
            candidates = anchors = frame
            candidate_indices = local_indices = torch.arange(
                self.num_points,
                device=frame.device,
                dtype=torch.long,
            )
            unique_count = self.num_points
            selection_mode = "first-direct"
            if event_after_voxel2 is not None:
                event_after_voxel2.record()
        elif n1 <= self.target_candidate_count:
            candidates = frame
            candidate_indices = torch.arange(
                n1, device=frame.device, dtype=torch.long
            )
            if event_after_voxel2 is not None:
                event_after_voxel2.record()
            anchors, local_indices = self._select_exact_count(candidates)
            unique_count = self.num_points
            selection_mode = f"first-{self.final_selection}"
        else:
            candidates, candidate_indices, second_used = self._second_downsample(
                frame
            )
            if event_after_voxel2 is not None:
                event_after_voxel2.record()
            n2 = int(candidates.shape[0])
            if n2 < self.num_points:
                anchors, local_indices, unique_count = self._repeat(candidates)
                selection_mode = "voxel2-repeat"
            elif n2 == self.num_points:
                anchors = candidates.contiguous()
                local_indices = torch.arange(
                    self.num_points,
                    device=frame.device,
                    dtype=torch.long,
                )
                unique_count = self.num_points
                selection_mode = "voxel2-direct"
            else:
                anchors, local_indices = self._select_exact_count(candidates)
                unique_count = self.num_points
                selection_mode = f"voxel2-{self.final_selection}"

        if event_after_exact is not None:
            event_after_exact.record()
            assert event_start is not None and event_after_voxel2 is not None
            timing_events = PreprocessTimingEvents(
                event_start,
                event_after_voxel2,
                event_after_exact,
            )

        final_indices = candidate_indices.index_select(
            0, local_indices
        ).contiguous()

        # These checks are metadata-only and do not synchronize CUDA.
        if anchors.shape != (self.num_points, 3):
            raise RuntimeError(
                f"Preprocessing produced {tuple(anchors.shape)}, expected "
                f"({self.num_points}, 3)."
            )
        if self.num_points < 1024:
            raise RuntimeError("DifFlow3D model input must contain >=1024 points.")

        if self.validate_finite and not collect_diagnostics:
            if not bool(torch.isfinite(anchors).all().item()):
                raise RuntimeError("Invalid final DifFlow3D model anchors.")

        anchor_ids = (
            point_ids.index_select(0, final_indices).contiguous()
            if point_ids is not None
            else None
        )
        info: dict[str, object] = {
            "input_count": n1,
            "candidate_count": int(candidates.shape[0]),
            "anchor_count": int(anchors.shape[0]),
            "unique_anchor_count": int(unique_count),
            "target_candidate_count": self.target_candidate_count,
            "second_voxel_used": bool(second_used),
            "second_mode": "auto" if second_used else "bypass",
            "second_voxel_resolution_m": (
                self.second_voxel_size_m if second_used else None
            ),
            "selection_mode": selection_mode,
            "final_selection": self.final_selection,
        }
        if collect_diagnostics:
            info.update(self._diagnostics(anchors))

        return PreparedModelInput(
            anchors,
            final_indices,
            anchor_ids,
            info,
            timing_events,
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
        return {
            "second": second,
            "spatial": self.spatial_calibration,
            "anchor_info": prepared.info,
        }
