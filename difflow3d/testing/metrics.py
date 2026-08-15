"""Benchmark metrics and CUDA timing helpers."""
import numpy as np
import torch

def cuda_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device}.")
    return (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(cuda_index(device))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def safe_mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(values.mean())


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": percentile(array, 90.0),
        "p95": percentile(array, 95.0),
        "p99": percentile(array, 99.0),
        "max": float(array.max()),
    }


def print_timing_row(name: str, stats: dict[str, float]) -> None:
    if not stats:
        return
    print(
        f"{name:24s} "
        f"mean={stats['mean']:8.3f} ms  "
        f"med={stats['median']:8.3f}  "
        f"p95={stats['p95']:8.3f}  "
        f"max={stats['max']:8.3f}"
    )

def metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": percentile(values, 95.0),
        "max": float(values.max()),
    }

