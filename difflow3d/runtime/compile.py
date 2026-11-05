"""Optional Inductor compilation of the DifFlow3D inference hot spots (point-wise op fusion).

Wraps, at class/module level, the eager functions that the streaming runner records into its CUDA graphs:
the recurrent GRU update, the bidirectional cross features and the flow embedding of the fast cross path, the
encoder's PointConv / PointConvD / level-0 context convolution and the point warping. Inductor fuses their
point-wise chains (bias, LeakyReLU, residual adds, concatenations, neighbourhood max) and lowers the 1x1
convolutions to TF32 GEMMs. It compiles without its own CUDA graphs ("-no-cudagraphs"); the runner's warm-up
runs trigger the compilation before the graphs are captured.

Real2 (RTX 5090, deployed preprocessing, dense points): predict() p50 7.69 -> 6.52 ms (max-autotune) / 6.82 ms
(default); EPE unchanged (2.9367 vs 2.9355 mm), single points differ by up to a few mm because the fused
arithmetic rounds differently (the model amplifies 1 um input changes to ~0.1 mm). Start-up: max-autotune
~210 s cold, ~50 s with a warm Inductor cache (TORCHINDUCTOR_CACHE_DIR); default ~60 s cold.
"""
from __future__ import annotations

import importlib

import torch

MODES = {"default": "default", "max-autotune": "max-autotune-no-cudagraphs"}
_TARGETS = (
    ("recurrent", "RecurrentUnit._bidirectional_fast"),
    ("recurrent", "RecurrentUnit._flow_embedding_fast"),
    ("recurrent", "DiffusionSceneFlowGRUResidual._gru_update_eval"),
    ("pointconv", "PointConvD.forward"),
    ("pointconv", "PointConv.forward"),
    ("pointconv", "PointWarping.forward"),
    ("encoder", "_pointconv_from_context"),
)
_applied: str | None = None


def enable_inductor(mode: str = "max-autotune") -> None:
    """Compile the hot spots with Inductor in ``mode`` (default | max-autotune). Idempotent per process."""
    global _applied
    if mode not in MODES:
        raise ValueError(f"compile mode must be one of {sorted(MODES)}, got {mode!r}")
    if _applied is not None:
        if _applied != mode:
            raise RuntimeError(f"DifFlow3D was already compiled with mode {_applied!r}")
        return
    import torch._dynamo

    # One compiled graph per module instance and bucket shape: well above Dynamo's default limit of 8.
    for name, value in (("cache_size_limit", 256), ("recompile_limit", 256), ("accumulated_recompile_limit", 4096)):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(int(getattr(torch._dynamo.config, name)), value))
    for module_name, attribute in _TARGETS:
        module = importlib.import_module(f"difflow3d.model.{module_name}")
        if "." in attribute:
            class_name, method = attribute.split(".")
            owner = getattr(module, class_name)
            name = method
        else:
            owner, name = module, attribute
        function = getattr(owner, name)
        if hasattr(function, "_torchdynamo_orig_callable"):
            continue  # already compiled (a previous call failed part-way)
        setattr(owner, name, torch.compile(function, mode=MODES[mode], dynamic=False, fullgraph=False))
    _applied = mode
