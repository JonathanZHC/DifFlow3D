"""Piecewise-constant canonical normalisation of the streaming input.

Model-space points are ``x_model = k (x_world - c) + c_model``: the anchor centroid ``c`` goes to the model
centre and the anchor RMS radius to ``model_radius`` (defaults: FT3D_s train medians at 8192 points), so any
object is seen at the scale and depth the checkpoint was trained on. One transform is shared by both frames of
a pair and held across the stream (the encoding of the previous frame is reused); it is re-anchored only when
the frame's RMS radius drifts by more than ``scale_ratio`` or its centroid by more than ``center_shift`` anchor
radii, and on ``reset()`` (a new stream). The runner then re-encodes the previous frame under the new transform.
"""
from __future__ import annotations

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
    ) -> None:
        if model_radius <= 0.0 or scale_ratio <= 1.0 or center_shift <= 0.0:
            raise ValueError("model_radius > 0, scale_ratio > 1 and center_shift > 0 are required")
        self.model_center = tuple(float(v) for v in model_center)
        self.model_radius = float(model_radius)
        self.scale_ratio = float(scale_ratio)
        self.center_shift = float(center_shift)
        self.per_frame = bool(per_frame)                 # re-anchor every frame (evaluation only)
        self.restage_previous = bool(restage_previous)   # False only to measure a broken stream (evaluation only)
        self.reanchor_count = 0
        self._anchor: tuple[torch.Tensor, float] | None = None
        self._transform: Transform | None = None

    def reset(self) -> None:
        self._anchor = None
        self._transform = None

    def update(self, points: torch.Tensor) -> tuple[Transform, bool]:
        """Transform for this frame and whether it differs from the previous one. One 4-value
        device -> host copy (centroid and RMS radius) per frame."""
        center = points.mean(0)
        stats = torch.cat([center, ((points - center) ** 2).sum(1).mean().sqrt().reshape(1)]).tolist()
        radius = max(float(stats[3]), 1.0e-6)
        if self._anchor is not None and not self.per_frame:
            anchor_center, anchor_radius = self._anchor
            drift = float(torch.linalg.norm(center - anchor_center))
            if max(radius / anchor_radius, anchor_radius / radius) <= self.scale_ratio and drift <= self.center_shift * anchor_radius:
                return self._transform, False
        changed = self._anchor is not None
        self._anchor = (center.clone(), radius)
        self._transform = Transform(center=center.clone(), scale=self.model_radius / radius,
                                    model_center=torch.tensor(self.model_center, device=points.device, dtype=points.dtype))
        self.reanchor_count += int(changed)
        return self._transform, changed
