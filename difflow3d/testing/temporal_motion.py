"""Streaming velocity-only temporal filtering for synthetic benchmarks.

The estimator keeps only the state required by the current deployment path:
raw first-order velocity when disabled, or a velocity-domain random-walk KF
when enabled. Previous KF states are reassociated to current anchors with the
track-aware CUDA transport operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from difflow3d.runtime import CudaAnchorTemporalOps


@dataclass(frozen=True)
class TemporalMotionEstimate:
    """Filtered anchor velocity and temporal-support information."""

    velocity: torch.Tensor
    support_counts: torch.Tensor


class StreamingAnchorMotionEstimator:
    """Stateful velocity-only KF for sparse anchors.

    With ``motion_estimation.kalman.enabled=false`` the benchmark bypasses this
    filter and uses raw ``flow / dt``. With the flag enabled, previous filtered
    velocity/covariance is transported strictly within persistent track IDs and
    updated by the fused CUDA KF kernel.
    """

    def __init__(self, config: dict, *, device: torch.device) -> None:
        motion = config.get("motion_estimation", {})
        kf_cfg = dict(motion.get("kalman", {}))
        self.kf_enabled = bool(kf_cfg.get("enabled", False))
        self.device = device

        allowed_motion = {"kalman"}
        unknown_motion = sorted(set(motion) - allowed_motion)
        if unknown_motion:
            raise ValueError(
                "Unknown motion_estimation settings: " + ", ".join(unknown_motion)
            )

        allowed_kf = {
            "enabled",
            "process_velocity_std_mps",
            "measurement_noise_std_mps",
            "initial_velocity_std_mps",
            "min_innovation_variance",
        }
        unknown_kf = sorted(set(kf_cfg) - allowed_kf)
        if unknown_kf:
            raise ValueError(
                "Unknown motion_estimation.kalman settings: "
                + ", ".join(unknown_kf)
            )

        # Cache scalar tuning parameters once; the per-frame hot path only
        # launches transport/KF work and does not repeatedly parse the config.
        self.process_velocity_std_mps = float(
            kf_cfg.get("process_velocity_std_mps", 0.05)
        )
        self.measurement_noise_std_mps = float(
            kf_cfg.get("measurement_noise_std_mps", 0.10)
        )
        self.initial_velocity_std_mps = float(
            kf_cfg.get("initial_velocity_std_mps", 0.30)
        )
        self.min_innovation_variance = float(
            kf_cfg.get("min_innovation_variance", 1.0e-6)
        )
        self.ops: CudaAnchorTemporalOps | None = None
        if self.kf_enabled:
            # Temporal reassociation uses the same Gaussian neighborhood as
            # dense recovery. Keep one source of truth in ``recovery`` so the
            # two spatial operators cannot silently drift apart.
            recovery = config.get("recovery", {})
            self.ops = CudaAnchorTemporalOps(
                softmax_sigma_m=float(recovery.get("softmax_sigma_m", 0.025)),
                local_radius_sigma=float(recovery.get("local_radius_sigma", 4.0)),
                local_hash_size_factor=float(
                    recovery.get("local_hash_size_factor", 4.0)
                ),
                # State transport is deliberately local-only. If no same-track
                # support exists in the radius, the KF treats the anchor as
                # unsupported instead of searching the whole previous frame.
                global_same_track_fallback=False,
            )

        self._previous_warped: torch.Tensor | None = None
        self._previous_track_ids: torch.Tensor | None = None

        # KF state: [vx, vy, vz, Pvx, Pvy, Pvz].
        self._state6: torch.Tensor | None = None
        # For spatial reassociation, transport E[v] and E[P + v^2] rather than
        # averaging P directly. This preserves uncertainty from local velocity
        # variation without adding another CUDA kernel.
        self._transport_moments6: torch.Tensor | None = None
        self._transport_mean_sq3: torch.Tensor | None = None
        self._filtered_warped: torch.Tensor | None = None
        self._zero_state6: torch.Tensor | None = None
        self._zero_support: torch.Tensor | None = None

    @property
    def mode_name(self) -> str:
        return "velocity_kalman" if self.kf_enabled else "first_order"

    def reset(self) -> None:
        self._previous_warped = None
        self._previous_track_ids = None
        # Keep allocated buffers for reuse; only history validity is reset.

    @staticmethod
    def _ensure_like(
        current: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if (
            current is None
            or current.shape != reference.shape
            or current.device != reference.device
            or current.dtype != reference.dtype
        ):
            return torch.empty_like(reference)
        return current

    def _save_history(
        self,
        *,
        warped_points: torch.Tensor,
        track_ids: torch.Tensor,
    ) -> None:
        self._previous_warped = self._ensure_like(
            self._previous_warped, warped_points
        )
        self._previous_warped.copy_(warped_points)

        track_ids = track_ids.contiguous().to(dtype=torch.int32)
        self._previous_track_ids = self._ensure_like(
            self._previous_track_ids, track_ids
        )
        self._previous_track_ids.copy_(track_ids)

    def _ensure_buffers(self, count: int) -> None:
        shape6 = (count, 6)
        if (
            self._state6 is None
            or self._state6.shape != shape6
            or self._state6.device != self.device
        ):
            self._state6 = torch.empty(
                shape6, device=self.device, dtype=torch.float32
            )
            self._transport_moments6 = torch.empty(
                shape6, device=self.device, dtype=torch.float32
            )
            self._transport_mean_sq3 = torch.empty(
                (count, 3), device=self.device, dtype=torch.float32
            )
            self._filtered_warped = torch.empty(
                (count, 3), device=self.device, dtype=torch.float32
            )
            self._zero_state6 = torch.zeros(
                shape6, device=self.device, dtype=torch.float32
            )
            self._zero_support = torch.zeros(
                (count,), device=self.device, dtype=torch.int32
            )

    @torch.inference_mode()
    def estimate(
        self,
        *,
        source_points: torch.Tensor,
        flow: torch.Tensor,
        track_ids: torch.Tensor,
        dt_s: float,
    ) -> TemporalMotionEstimate:
        """Update the velocity-only KF for the current anchor flow pair."""
        if not self.kf_enabled or self.ops is None:
            raise RuntimeError("estimate() requires motion_estimation.kalman.enabled=true")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        flow = flow.contiguous().float()
        source_points = source_points.contiguous().float()
        track_ids = track_ids.contiguous().to(dtype=torch.int32)
        count = int(flow.shape[0])
        self._ensure_buffers(count)

        assert self._state6 is not None
        assert self._transport_moments6 is not None
        assert self._transport_mean_sq3 is not None
        assert self._filtered_warped is not None
        assert self._zero_state6 is not None
        assert self._zero_support is not None

        has_history = (
            self._previous_warped is not None
            and self._previous_track_ids is not None
        )
        if has_history:
            # Moment-match the local mixture after spatial transport:
            #   m1 = E[v], m2 = E[P + v^2], P = m2 - m1^2.
            moments6 = self._transport_moments6
            moments6[:, :3].copy_(self._state6[:, :3])
            torch.mul(
                self._state6[:, :3],
                self._state6[:, :3],
                out=moments6[:, 3:6],
            )
            moments6[:, 3:6].add_(self._state6[:, 3:6])

            history = self.ops.transport(
                query_points=source_points,
                source_points=self._previous_warped,
                source_values=moments6,
                query_track_ids=track_ids,
                source_track_ids=self._previous_track_ids,
            )
            previous_state6 = history.values
            torch.mul(
                previous_state6[:, :3],
                previous_state6[:, :3],
                out=self._transport_mean_sq3,
            )
            previous_state6[:, 3:6].sub_(self._transport_mean_sq3)
            previous_state6[:, 3:6].clamp_(min=1.0e-12, max=1.0e12)
            support_counts = history.support_counts
        else:
            previous_state6 = self._zero_state6
            support_counts = self._zero_support

        result = self.ops.kalman_update(
            current_flow=flow,
            transported_previous_state6=previous_state6,
            support_counts=support_counts,
            dt_s=float(dt_s),
            process_velocity_std_mps=self.process_velocity_std_mps,
            measurement_noise_std_mps=self.measurement_noise_std_mps,
            initial_velocity_std_mps=self.initial_velocity_std_mps,
            min_innovation_variance=self.min_innovation_variance,
            output_state6=self._state6,
        )

        # Keep the state spatially attached to the filtered target position.
        self._filtered_warped.copy_(result.velocity).mul_(float(dt_s))
        self._filtered_warped.add_(source_points)
        self._save_history(
            warped_points=self._filtered_warped,
            track_ids=track_ids,
        )

        return TemporalMotionEstimate(
            velocity=result.velocity,
            support_counts=result.support_counts,
        )
