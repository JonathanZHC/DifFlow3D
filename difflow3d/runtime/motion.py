"""CUDA anchor-level temporal motion estimation for ScenePredictor integration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from difflow3d.ops.pointnet2 import pointnet2_utils


@dataclass(frozen=True)
class AnchorTransportResult:
    values: torch.Tensor
    support_counts: torch.Tensor


@dataclass(frozen=True)
class KalmanResult:
    """Velocity-only KF result.

    ``state6`` stores ``[vx, vy, vz, Pvx, Pvy, Pvz]``. There is no
    acceleration state. Every finite velocity measurement is fused by the
    standard Kalman update.
    """

    velocity: torch.Tensor
    state6: torch.Tensor
    support_counts: torch.Tensor


@dataclass
class _TransportBuffers:
    anchor_count: int
    query_count: int
    channels: int
    hash_size: int
    hash_heads: torch.Tensor
    anchor_next: torch.Tensor
    anchor_cells: torch.Tensor
    output: torch.Tensor
    support_counts: torch.Tensor


class CudaAnchorTemporalOps:
    """Low-overhead CUDA operators for sparse anchor temporal estimation.

    The class reuses hash/output buffers across frames when tensor sizes stay
    unchanged. Spatial transport is always strict same-track. Temporal kernels
    operate only on sparse anchors and never perform per-anchor Python loops.
    """

    _VALID_CHANNELS = (3, 6)

    def __init__(
        self,
        *,
        softmax_sigma_m: float = 0.025,
        local_radius_sigma: float = 4.0,
        local_hash_size_factor: float = 4.0,
        global_same_track_fallback: bool = False,
    ) -> None:
        if softmax_sigma_m <= 0.0:
            raise ValueError("softmax_sigma_m must be positive")
        if local_radius_sigma <= 0.0:
            raise ValueError("local_radius_sigma must be positive")
        if local_hash_size_factor < 1.0:
            raise ValueError("local_hash_size_factor must be >= 1.0")
        if not pointnet2_utils.has_gaussian_softmax_transport_local_track_aware():
            raise RuntimeError(
                "Temporal state transport is unavailable. Rebuild with: "
                "bash scripts/build_pointnet2_ops.sh"
            )
        if not pointnet2_utils.has_anchor_velocity_kalman_op():
            raise RuntimeError(
                "Anchor velocity-KF CUDA kernel is unavailable. Rebuild with: "
                "bash scripts/build_pointnet2_ops.sh"
            )

        self.softmax_sigma_m = float(softmax_sigma_m)
        self.local_radius_sigma = float(local_radius_sigma)
        self.local_hash_size_factor = float(local_hash_size_factor)
        self.global_same_track_fallback = bool(global_same_track_fallback)
        self._buffers: dict[tuple[str, int, int, int, int], _TransportBuffers] = {}

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (int(value) - 1).bit_length()

    @staticmethod
    def _as_float_matrix(tensor: torch.Tensor, columns: int, name: str) -> torch.Tensor:
        if tensor.ndim != 2 or tensor.shape[1] != columns:
            raise ValueError(f"{name} must be [N,{columns}]")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
        return tensor.contiguous().float()

    @staticmethod
    def _as_track_ids(tensor: torch.Tensor, count: int, device: torch.device, name: str) -> torch.Tensor:
        if tensor.ndim != 1 or tensor.shape[0] != count:
            raise ValueError(f"{name} must be [{count}]")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
        return tensor.contiguous().to(dtype=torch.int32)

    def _get_buffers(
        self,
        *,
        device: torch.device,
        anchor_count: int,
        query_count: int,
        channels: int,
    ) -> _TransportBuffers:
        device_index = -1 if device.index is None else int(device.index)
        key = (device.type, device_index, anchor_count, query_count, channels)
        buffers = self._buffers.get(key)
        if buffers is not None:
            return buffers

        hash_size = self._next_power_of_two(
            max(32, math.ceil(anchor_count * self.local_hash_size_factor))
        )
        buffers = _TransportBuffers(
            anchor_count=anchor_count,
            query_count=query_count,
            channels=channels,
            hash_size=hash_size,
            hash_heads=torch.empty((hash_size,), device=device, dtype=torch.int32),
            anchor_next=torch.empty((anchor_count,), device=device, dtype=torch.int32),
            anchor_cells=torch.empty((anchor_count, 3), device=device, dtype=torch.int32),
            output=torch.empty((query_count, channels), device=device, dtype=torch.float32),
            support_counts=torch.empty((query_count,), device=device, dtype=torch.int32),
        )
        self._buffers[key] = buffers
        return buffers

    @torch.inference_mode()
    def transport(
        self,
        *,
        query_points: torch.Tensor,
        source_points: torch.Tensor,
        source_values: torch.Tensor,
        query_track_ids: torch.Tensor,
        source_track_ids: torch.Tensor,
    ) -> AnchorTransportResult:
        """Transport previous state to current source anchors.

        ``source_points`` are previous warped anchors, while ``query_points``
        are current source anchors. A zero support count means no valid history.
        """
        queries = self._as_float_matrix(query_points, 3, "query_points")
        sources = self._as_float_matrix(source_points, 3, "source_points")
        if source_values.ndim != 2 or source_values.shape[0] != sources.shape[0]:
            raise ValueError("source_values must be [K,C]")
        channels = int(source_values.shape[1])
        if channels not in self._VALID_CHANNELS:
            raise ValueError("source_values must have 3 or 6 channels")
        if source_values.device != queries.device or sources.device != queries.device:
            raise ValueError("all transport tensors must be on the same CUDA device")
        values = source_values.contiguous().float()
        qids = self._as_track_ids(
            query_track_ids, int(queries.shape[0]), queries.device, "query_track_ids"
        )
        sids = self._as_track_ids(
            source_track_ids, int(sources.shape[0]), sources.device, "source_track_ids"
        )

        buffers = self._get_buffers(
            device=queries.device,
            anchor_count=int(sources.shape[0]),
            query_count=int(queries.shape[0]),
            channels=channels,
        )
        radius = self.local_radius_sigma * self.softmax_sigma_m
        pointnet2_utils.pointnet2.gaussian_recovery_hash_build_wrapper(
            sources,
            float(radius),
            buffers.hash_heads,
            buffers.anchor_next,
            buffers.anchor_cells,
        )
        pointnet2_utils.pointnet2.gaussian_softmax_transport_local_track_aware_wrapper(
            queries,
            sources,
            values,
            qids,
            sids,
            self.softmax_sigma_m,
            float(radius),
            float(radius),
            self.global_same_track_fallback,
            buffers.hash_heads,
            buffers.anchor_next,
            buffers.anchor_cells,
            buffers.output,
            buffers.support_counts,
        )
        return AnchorTransportResult(
            values=buffers.output,
            support_counts=buffers.support_counts,
        )

    @torch.inference_mode()
    def kalman_update(
        self,
        *,
        current_flow: torch.Tensor,
        transported_previous_state6: torch.Tensor,
        support_counts: torch.Tensor,
        dt_s: float,
        process_velocity_std_mps: float = 0.05,
        measurement_noise_std_mps: float = 0.10,
        initial_velocity_std_mps: float = 0.30,
        min_innovation_variance: float = 1.0e-6,
        output_state6: torch.Tensor | None = None,
    ) -> KalmanResult:
        """Run one fused velocity-only KF update on current anchors.

        The state is ``[v, diag(P_v)]`` with a velocity random-walk model,
        ``v_k^- = v_{k-1}`` and ``P_k^- = P_{k-1} + Q_v``. Every finite
        DifFlow velocity measurement is fused; non-finite measurements use the
        finite prediction as a numerical-safety fallback.
        """
        flow = self._as_float_matrix(current_flow, 3, "current_flow")
        previous = self._as_float_matrix(
            transported_previous_state6, 6, "transported_previous_state6"
        )
        if previous.shape[0] != flow.shape[0] or previous.device != flow.device:
            raise ValueError("transported_previous_state6 must match current_flow")
        support = self._as_track_ids(
            support_counts, int(flow.shape[0]), flow.device, "support_counts"
        )
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if process_velocity_std_mps < 0.0:
            raise ValueError("process_velocity_std_mps must be non-negative")
        if measurement_noise_std_mps <= 0.0:
            raise ValueError("measurement_noise_std_mps must be positive")
        if initial_velocity_std_mps <= 0.0:
            raise ValueError("initial_velocity_std_mps must be positive")
        if min_innovation_variance <= 0.0:
            raise ValueError("min_innovation_variance must be positive")

        count = int(flow.shape[0])
        if output_state6 is None:
            state6 = torch.empty((count, 6), device=flow.device, dtype=torch.float32)
        else:
            state6 = self._as_float_matrix(output_state6, 6, "output_state6")
            if state6.shape[0] != count or state6.device != flow.device:
                raise ValueError("output_state6 must match current_flow")

        pointnet2_utils.pointnet2.anchor_kalman_update_wrapper(
            flow,
            previous,
            support,
            float(dt_s),
            float(process_velocity_std_mps) ** 2,
            float(measurement_noise_std_mps) ** 2,
            float(initial_velocity_std_mps) ** 2,
            float(min_innovation_variance),
            state6,
        )
        return KalmanResult(
            velocity=state6[:, :3],
            state6=state6,
            support_counts=support,
        )

