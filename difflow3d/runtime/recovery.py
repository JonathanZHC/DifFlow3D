"""Dense Gaussian-softmax motion recovery in real/world metric coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from difflow3d.ops.pointnet2 import pointnet2_utils


@dataclass(frozen=True)
class DenseMotionRecovery:
    flow: torch.Tensor
    velocity: torch.Tensor
    # Present only for the sparse local backend. A zero count means the query
    # used the exact-global fallback because no anchor fell inside the cutoff.
    local_neighbor_counts: torch.Tensor | None = None


class SoftmaxAnchorMotionRecoverer:
    """Recover dense world-space flow from sparse anchors.

    ``softmax_sigma_m`` is always expressed in real/world metres.

    Backends:
      * ``global``: exact all-anchor warp-per-query CUDA kernel.
      * ``local``: radius-truncated CUDA Gaussian softmax over an anchor hash
        grid. Optional query/anchor track IDs enable one-call same-track
        conditioning. Queries without an in-radius anchor fall back globally.
      * ``torch``: chunked exact PyTorch reference/fallback.
      * ``auto``: ``global`` when the CUDA extension is available, otherwise
        ``torch``. ``auto`` never silently selects the approximate local path.

    Velocity is derived as ``recovered_flow / dt``; anchor velocity is not
    interpolated independently because it is exactly flow/dt in this pipeline.
    """

    _VALID_BACKENDS = {"auto", "global", "local", "torch"}

    def __init__(
        self,
        *,
        chunk_size: int,
        softmax_sigma_m: float,
        backend: str = "auto",
        local_radius_sigma: float = 4.0,
        local_hash_size_factor: float = 4.0,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if softmax_sigma_m <= 0.0:
            raise ValueError("softmax_sigma_m must be positive.")
        if local_radius_sigma <= 0.0:
            raise ValueError("local_radius_sigma must be positive.")
        if local_hash_size_factor < 1.0:
            raise ValueError("local_hash_size_factor must be >= 1.0.")

        backend = str(backend).lower()
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                "recovery backend must be one of: auto, global, local, torch"
            )

        if backend == "auto":
            backend = (
                "global"
                if pointnet2_utils.has_gaussian_softmax_recovery()
                else "torch"
            )
        elif backend == "global":
            self._require_cuda_backend(
                "global", pointnet2_utils.has_gaussian_softmax_recovery()
            )
        elif backend == "local":
            self._require_cuda_backend(
                "local", pointnet2_utils.has_gaussian_softmax_recovery_local()
            )

        self.chunk_size = int(chunk_size)
        self.softmax_sigma_m = float(softmax_sigma_m)
        self.backend = backend
        self.local_radius_sigma = float(local_radius_sigma)
        self.local_hash_size_factor = float(local_hash_size_factor)

    @staticmethod
    def _require_cuda_backend(backend: str, available: bool) -> None:
        if available:
            return
        raise RuntimeError(
            f"Recovery backend {backend!r} was requested but the required "
            "pointnet2_cuda kernel is unavailable. Run: "
            "bash scripts/build_pointnet2_ops.sh"
        )

    @torch.inference_mode()
    def _recover_torch(
        self,
        queries: torch.Tensor,
        anchors: torch.Tensor,
        anchor_flow: torch.Tensor,
    ) -> torch.Tensor:
        anchor_t = anchors.transpose(0, 1).contiguous()
        anchor_norm2 = anchors.square().sum(dim=1).unsqueeze(0)
        sigma2 = self.softmax_sigma_m * self.softmax_sigma_m
        anchor_bias = -0.5 * anchor_norm2 / sigma2
        recovered = torch.empty_like(queries)

        # ||q-a||² = ||q||² + ||a||² - 2q·a. The query-only term cancels
        # inside the softmax, which lets each chunk use one GEMM + softmax.
        for start in range(0, queries.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, queries.shape[0])
            query = queries[start:end]
            logits = torch.addmm(
                anchor_bias.expand(query.shape[0], -1),
                query,
                anchor_t,
                beta=1.0,
                alpha=1.0 / sigma2,
            )
            recovered[start:end] = torch.softmax(logits, dim=1) @ anchor_flow
        return recovered

    @staticmethod
    def _validate_inputs(
        query_points: torch.Tensor,
        anchor_points: torch.Tensor,
        anchor_flow: torch.Tensor,
        dt_s: float,
    ) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError("query_points must be [Q,3].")
        if anchor_points.ndim != 2 or anchor_points.shape[1] != 3:
            raise ValueError("anchor_points must be [K,3].")
        if anchor_flow.shape != anchor_points.shape:
            raise ValueError("anchor_flow must match anchor_points.")
        if not (
            query_points.device == anchor_points.device == anchor_flow.device
        ):
            raise ValueError("Recovery tensors must be on the same device.")

    @torch.inference_mode()
    def recover(
        self,
        *,
        query_points: torch.Tensor,
        anchor_points: torch.Tensor,
        anchor_flow: torch.Tensor,
        dt_s: float,
        query_track_ids: torch.Tensor | None = None,
        anchor_track_ids: torch.Tensor | None = None,
    ) -> DenseMotionRecovery:
        self._validate_inputs(query_points, anchor_points, anchor_flow, dt_s)
        if self.backend != "torch" and query_points.device.type != "cuda":
            raise ValueError(f"Recovery backend {self.backend!r} requires CUDA tensors.")

        queries = query_points.contiguous().float()
        anchors = anchor_points.contiguous().float()
        flow = anchor_flow.contiguous().float()
        backend = self.backend
        local_counts: torch.Tensor | None = None

        track_aware = query_track_ids is not None or anchor_track_ids is not None
        if track_aware:
            if query_track_ids is None or anchor_track_ids is None:
                raise ValueError(
                    "query_track_ids and anchor_track_ids must be provided together"
                )
            if backend != "local":
                raise ValueError(
                    "track-aware recovery currently requires backend='local'"
                )

        if backend == "global":
            recovered_flow = pointnet2_utils.gaussian_softmax_recovery(
                queries, anchors, flow, self.softmax_sigma_m
            )
        elif backend == "local":
            if track_aware:
                recovered_flow, local_counts = (
                    pointnet2_utils.gaussian_softmax_recovery_local_track_aware(
                        queries,
                        anchors,
                        flow,
                        query_track_ids,
                        anchor_track_ids,
                        self.softmax_sigma_m,
                        radius_sigma=self.local_radius_sigma,
                        hash_size_factor=self.local_hash_size_factor,
                    )
                )
            else:
                recovered_flow, local_counts = (
                    pointnet2_utils.gaussian_softmax_recovery_local(
                        queries,
                        anchors,
                        flow,
                        self.softmax_sigma_m,
                        radius_sigma=self.local_radius_sigma,
                        hash_size_factor=self.local_hash_size_factor,
                    )
                )
        else:
            recovered_flow = self._recover_torch(queries, anchors, flow)

        return DenseMotionRecovery(
            flow=recovered_flow.contiguous(),
            velocity=(recovered_flow / float(dt_s)).contiguous(),
            local_neighbor_counts=local_counts,
        )
