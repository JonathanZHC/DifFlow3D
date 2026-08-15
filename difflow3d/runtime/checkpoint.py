"""Checkpoint loading for inference-only DifFlow3D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointReport:
    is_exact: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint format: {type(checkpoint).__name__}."
        )
    for key in ("state_dict", "model_state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and nested:
            return nested
    return checkpoint


def _strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool,
) -> CheckpointReport:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DifFlow3D checkpoint not found: {checkpoint_path}"
        )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_extract_state_dict(raw))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    return CheckpointReport(
        is_exact=not missing and not unexpected,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
