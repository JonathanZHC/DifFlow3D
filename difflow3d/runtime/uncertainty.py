"""Per-point covariance of the predicted scene flow (Cholesky head on the fine-level diffusion features).

Port of the deployed U3 head (``NormCovHead`` of the human-motion tools, arch ``norm`` without input normalisation or
metric output, displacement-scaled with a piecewise-linear ``s(|flow|)``), restricted to that configuration.

Inputs, channels-last and in model units:
  * ``xyz``  ``[B,N,3]``   source points;
  * ``feat`` ``[B,N,128]`` condition of the last fine-level denoiser call: cat(GRU latent passed into the call (64),
    encoder features (32), flow embedding (32)), see ``RecurrentUnit.capture_head_inputs``;
  * ``u``    ``[B,N,1]``   certainty returned by that call;
  * ``flow`` ``[B,N,3]``   final flow;
  * ``kk``   ``[B]``       model units per metre (canonical scale); the runner passes it by keyword.
Output: lower-triangular ``L`` ``[B,N,3,3]`` in model units, ``Sigma_model = L L^T``, ``Sigma_world = Sigma_model / kk^2``.
The forward pass has static shapes and no host synchronisation, so it runs inside the runner's decode CUDA graphs.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflow3d.model.pointconv import knn_point

# |predicted flow| knots (mm) of the piecewise-linear log s(log(|flow| + 0.5)); the end slopes are extrapolated.
PWL_KNOTS_MM = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)

# Checkpoint fields this port requires, and the head variants it does not implement.
_REQUIRED_META = {"arch": "norm", "feat_dim": 128, "k": 16, "norm_in": False, "phys": False, "disp_form": "pwl"}
_UNSUPPORTED_FLAGS = ("disp_learn", "no_flow_in")


def _gather(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``[B,N,C]`` values at ``[B,N,K]`` indices -> ``[B,N,K,C]``."""
    batch = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1)
    return values[batch, idx]


class FlowCovarianceHead(nn.Module):
    """Cholesky factor of the flow error covariance, relative to a displacement-only scale ``s(|flow|)``.

    A point MLP on [feat, u, flow, log k-th neighbour distance] and an edge MLP over the k nearest source points
    (self included, relative positions, max-pooled) predict a relative factor ``L_rel`` (diagonal
    ``min_rel + sigma_rel0 softplus / ln 2``, off-diagonal ``sigma_rel0 raw``); the result is ``s * L_rel``.
    """

    def __init__(
        self,
        *,
        disp_params: tuple[float, ...] | list[float],
        feat_dim: int = 128,
        hidden: int = 128,
        k: int = 16,
        sigma_rel0: float = 1.0,
        min_rel: float = 0.05,
    ) -> None:
        super().__init__()
        if len(disp_params) != len(PWL_KNOTS_MM):
            raise ValueError(f"disp_params needs {len(PWL_KNOTS_MM)} values (log s at PWL_KNOTS_MM), got {len(disp_params)}.")
        self.feat_dim = int(feat_dim)
        self.k = int(k)
        self.sigma_rel0 = float(sigma_rel0)
        self.min_rel = float(min_rel)
        self.point_mlp = nn.Sequential(nn.Linear(self.feat_dim + 1 + 3 + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.edge_mlp = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 6))
        # Fixed displacement scale (fitted on the training data, not part of the state dict).
        knots = torch.log(torch.tensor(PWL_KNOTS_MM, dtype=torch.float64) + 0.5).float()
        self.register_buffer("pwl_log_knots", knots, persistent=False)
        self.register_buffer("pwl_log_scale", torch.tensor([float(v) for v in disp_params]), persistent=False)

    def displacement_scale(self, flow: torch.Tensor, kk: torch.Tensor) -> torch.Tensor:
        """Per-axis std of the displacement-only error model, ``[B,N]`` model units: ``s_mm(|flow| mm) * kk / 1000``."""
        kk = kk.reshape(-1, 1).float()
        log_disp = torch.log((flow.norm(dim=-1) * (1000.0 / kk)).clamp_min(0.0) + 0.5)
        knots = self.pwl_log_knots
        i = torch.searchsorted(knots, log_disp.contiguous()).clamp(1, knots.numel() - 1)
        u0, u1 = knots[i - 1], knots[i]
        t0, t1 = self.pwl_log_scale[i - 1], self.pwl_log_scale[i]
        return torch.exp(t0 + (t1 - t0) * (log_disp - u0) / (u1 - u0)) * (kk / 1000.0)

    def forward(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        u: torch.Tensor,
        flow: torch.Tensor,
        kk: torch.Tensor,
        idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``L`` ``[B,N,3,3]`` (model units); ``idx`` = optional precomputed ``[B,N,k]`` self-kNN (self included)."""
        xyz = xyz.float()
        flow = flow.float()
        if idx is None:
            idx = knn_point(self.k, xyz.contiguous(), xyz.contiguous())
        rel = _gather(xyz, idx) - xyz.unsqueeze(2)
        # k-th neighbour distance = the largest of the k (order-free, so any exact kNN ordering gives the same input).
        spacing = rel.norm(dim=-1).amax(dim=-1, keepdim=True)
        h = self.point_mlp(torch.cat([feat.float(), u.float(), flow, torch.log(spacing + 1.0e-4)], dim=-1))
        e = self.edge_mlp(torch.cat([_gather(h, idx), rel], dim=-1)).amax(dim=2)
        raw = self.out(torch.cat([h, e], dim=-1))
        diag = self.min_rel + self.sigma_rel0 * F.softplus(raw[..., :3]) / math.log(2.0)
        off = self.sigma_rel0 * raw[..., 3:]
        zero = torch.zeros_like(diag[..., 0])
        chol = torch.stack(
            (diag[..., 0], zero, zero, off[..., 0], diag[..., 1], zero, off[..., 1], off[..., 2], diag[..., 2]), dim=-1
        ).view(*diag.shape[:-1], 3, 3)
        return chol * self.displacement_scale(flow, kk)[..., None, None]


def load_flow_covariance_head(path: str | Path, device: torch.device | str) -> tuple[FlowCovarianceHead, dict]:
    """Head checkpoint (dict with the training config and ``state_dict``) -> (``head.eval()`` on ``device``, meta).

    ``meta`` is the checkpoint without ``state_dict``; it includes ``iters`` (backbone iterations coarse,middle,fine the
    head was trained behind), ``backbone``, ``disp_form`` and ``seed``.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Uncertainty head checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise TypeError(f"{path} is not a head checkpoint (dict with config and 'state_dict').")
    meta = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    for key, expected in _REQUIRED_META.items():
        if meta.get(key) != expected:
            raise ValueError(f"{path}: {key}={meta.get(key)!r}, this head port supports only {key}={expected!r}.")
    for key in _UNSUPPORTED_FLAGS:
        if meta.get(key, False):
            raise ValueError(f"{path}: {key}=True is not supported by this head port.")
    for key in ("iters", "backbone", "seed", "hidden", "sigma_rel0", "min_rel", "disp_params"):
        if key not in meta:
            raise KeyError(f"{path}: checkpoint field {key!r} is missing.")
    head = FlowCovarianceHead(
        disp_params=meta["disp_params"],
        feat_dim=int(meta["feat_dim"]),
        hidden=int(meta["hidden"]),
        k=int(meta["k"]),
        sigma_rel0=float(meta["sigma_rel0"]),
        min_rel=float(meta["min_rel"]),
    )
    state = dict(checkpoint["state_dict"])
    # Input z-score statistics of NormCovHead, used only with norm_in.
    state.pop("in_mu", None)
    state.pop("in_sd", None)
    head.load_state_dict(state, strict=True)
    return head.to(device).eval(), meta
