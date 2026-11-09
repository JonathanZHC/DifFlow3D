"""Piecewise-constant canonical normalisation of the streaming input.

Model-space points are ``x_model = k (x_world - c) + c_model``: the anchor centroid ``c`` goes to the model
centre and the anchor RMS radius to ``model_radius`` (defaults: FT3D_s train medians at 8192 points), so any
object is seen at the scale and depth the checkpoint was trained on. One transform is shared by both frames of
a pair and held across the stream (the encoding of the previous frame is reused); it is re-anchored only when
the frame's RMS radius drifts by more than ``scale_ratio`` or its centroid by more than ``center_shift`` anchor
radii, and on ``reset()`` (a new stream). The runner then re-encodes the previous frame under the new transform.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

FT3D_MODEL_CENTER = (-0.52, -1.57, 21.36)     # metres, FT3D_s train median centroid (8192 points)
FT3D_MODEL_RADIUS = 8.97                      # metres, FT3D_s train median RMS radius


@dataclass(frozen=True)
class Transform:
    center: torch.Tensor          # [3] world anchor centroid
    scale: float                  # k = model_radius / anchor radius
    model_center: torch.Tensor    # [3]

    def to_model(self, points: torch.Tensor) -> torch.Tensor:
        return (points - self.center) * self.scale + self.model_center


class AnchoredNormalization:
    def __init__(
        self,
        *,
        model_center: tuple[float, float, float] = FT3D_MODEL_CENTER,
        model_radius: float = FT3D_MODEL_RADIUS,
        scale_ratio: float = 1.5,
        center_shift: float = 0.3,
        per_frame: bool = False,
        restage_previous: bool = True,
        mode: str = "canonical",
        target_spacing: float = 0.1,
        extent_min: float = 0.6,
        extent_max: float = 10.0,
        k_max: float | None = None,
        min_points: int = 0,
    ) -> None:
        """mode 'canonical': k = model_radius / RMS radius (every object at the FT3D radius). mode 'density': k makes the
        median nearest-neighbour spacing of the anchors equal target_spacing model units, then the cloud diameter is clamped
        to [extent_min, extent_max] (small sparse objects were magnified ~200x by 'canonical' and the network failed:
        SafetyFilterBench tools/difflow_sphere_study.py). k_max caps k in either mode. min_points > 0: a frame with fewer
        points raises (too few points for per-point scene flow; such a track needs a rigid / centroid estimate instead)."""
        if model_radius <= 0.0 or scale_ratio <= 1.0 or center_shift <= 0.0:
            raise ValueError("model_radius > 0, scale_ratio > 1 and center_shift > 0 are required")
        if mode not in ("canonical", "density"):
            raise ValueError(f"normalization mode must be 'canonical' or 'density', got {mode!r}")
        if target_spacing <= 0.0 or extent_min <= 0.0 or extent_max <= extent_min:
            raise ValueError("target_spacing > 0 and 0 < extent_min < extent_max are required")
        self.mode = str(mode)
        self.target_spacing = float(target_spacing)
        self.extent_min, self.extent_max = float(extent_min), float(extent_max)
        self.k_max = None if k_max is None else float(k_max)
        self.min_points = int(min_points)
        self.model_center = tuple(float(v) for v in model_center)
        self.model_radius = float(model_radius)
        self.scale_ratio = float(scale_ratio)
        self.center_shift = float(center_shift)
        self.per_frame = bool(per_frame)                 # re-anchor every frame (evaluation only)
        self.restage_previous = bool(restage_previous)   # False only to measure a broken stream (evaluation only)
        self.reanchor_count = 0
        self._anchor: tuple[tuple[float, float, float], float, float] | None = None   # host centroid, RMS radius, k basis
        self.last_scale_info: dict | None = None
        self._transform: Transform | None = None
        self._model_center: torch.Tensor | None = None   # device copy of model_center (per device and dtype)

    def reset(self) -> None:
        self._anchor = None
        self._transform = None

    def update(self, points: torch.Tensor) -> tuple[Transform, bool]:
        """Transform for this frame and whether it differs from the previous one. One 4-value
        device -> host copy (centroid and RMS radius) per frame; the re-anchor test runs on the host."""
        n = int(points.shape[0])
        if self.min_points > 0 and n < self.min_points:
            raise ValueError(f"DifFlow3D normalisation: {n} points < min_points {self.min_points}: too few points for per-point "
                             "scene flow (tiny / far object); this track needs a rigid or centroid velocity estimate instead")
        center = points.mean(0)
        stats = torch.cat([center, ((points - center) ** 2).sum(1).mean().sqrt().reshape(1)]).tolist()
        host_center, radius = tuple(stats[:3]), max(stats[3], 1.0e-6)
        # the quantity k is based on: the RMS radius (canonical) or the median nearest-neighbour spacing (density)
        if self.mode == "density":
            sub = points if n <= 2048 else points[torch.randperm(n, device=points.device)[:2048]]
            d = torch.cdist(sub, sub)
            d.fill_diagonal_(float("inf"))
            basis = max(float(d.min(1).values.median()), 1.0e-6)
            extent = max(2.0 * float((points - center).norm(dim=1).max()), 1.0e-6)
        else:
            basis, extent = radius, 0.0
        if self._anchor is not None and not self.per_frame:
            anchor_center, anchor_radius, anchor_basis = self._anchor
            drift = math.dist(host_center, anchor_center)
            if max(basis / anchor_basis, anchor_basis / basis) <= self.scale_ratio and drift <= self.center_shift * anchor_radius:
                return self._transform, False
        changed = self._anchor is not None
        self._anchor = (host_center, radius, basis)
        model_center = self._model_center
        if model_center is None or model_center.device != points.device or model_center.dtype != points.dtype:
            model_center = torch.tensor(self.model_center, device=points.device, dtype=points.dtype)
            self._model_center = model_center
        if self.mode == "density":
            k = self.target_spacing / basis
            k = max(k, self.extent_min / extent)            # the cloud must not be smaller than the network's receptive scale
            k = min(k, self.extent_max / extent)            # nor larger than a training scene
        else:
            k = self.model_radius / radius
        if self.k_max is not None:
            k = min(k, self.k_max)
        self.last_scale_info = {"n": n, "radius_m": radius, "spacing_m": basis if self.mode == "density" else None, "k": k,
                                "extent_model": extent * k if self.mode == "density" else 2.0 * radius * k}
        self._transform = Transform(center=center, scale=k, model_center=model_center)
        self.reanchor_count += int(changed)
        return self._transform, changed
