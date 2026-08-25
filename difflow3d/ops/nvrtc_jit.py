"""Minimal NVRTC + CUDA driver-API JIT helper (ctypes only).

Compiles CUDA C source to a cubin for the current device and launches kernels on
PyTorch's current stream (also valid inside ``torch.cuda.graph`` capture: the
launch is stream-ordered and no allocations happen here). Uses the ``libnvrtc``
bundled with pip-installed torch, so no ``nvcc`` is required at runtime.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys

import torch


def _find_nvrtc() -> str:
    candidates: list[str] = []
    for entry in sys.path:
        if entry and os.path.isdir(entry):
            candidates += glob.glob(os.path.join(entry, "nvidia", "**", "libnvrtc.so.*"), recursive=True)
    candidates += glob.glob("/usr/local/cuda*/lib64/libnvrtc.so.*")
    candidates = [c for c in candidates if "builtins" not in c and ".alt." not in c]
    if not candidates:
        raise RuntimeError("libnvrtc not found (needed for CUDA JIT kernels)")
    candidates.sort(key=os.path.basename, reverse=True)
    return candidates[0]


_nvrtc: ctypes.CDLL | None = None
_cuda: ctypes.CDLL | None = None


def _libs() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    global _nvrtc, _cuda
    if _nvrtc is None:
        _nvrtc = ctypes.CDLL(_find_nvrtc())
        _cuda = ctypes.CDLL("libcuda.so.1")
    return _nvrtc, _cuda  # type: ignore[return-value]


def _check_nvrtc(nvrtc: ctypes.CDLL, result: int, prog=None) -> None:
    if result != 0:
        message = f"nvrtc error {result}"
        if prog is not None:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(prog, buffer)
            message += "\n" + buffer.value.decode(errors="replace")
        raise RuntimeError(message)


def _check_cu(cuda: ctypes.CDLL, result: int) -> None:
    if result != 0:
        text = ctypes.c_char_p()
        cuda.cuGetErrorString(result, ctypes.byref(text))
        raise RuntimeError(f"cuda driver error {result}: {text.value}")


class Kernel:
    def __init__(self, cuda: ctypes.CDLL, fn: ctypes.c_void_p) -> None:
        self._cuda = cuda
        self._fn = fn

    def launch(self, grid, block, *args, shmem: int = 0, stream: int | None = None) -> None:
        """args: torch.Tensor -> pointer, int -> int32, float -> float32."""
        cargs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                cargs.append(ctypes.c_void_p(arg.data_ptr()))
            elif isinstance(arg, bool) or isinstance(arg, int):
                cargs.append(ctypes.c_int(int(arg)))
            elif isinstance(arg, float):
                cargs.append(ctypes.c_float(arg))
            else:
                raise TypeError(f"unsupported kernel argument type {type(arg)!r}")
        pointers = (ctypes.c_void_p * len(cargs))(
            *[ctypes.cast(ctypes.pointer(c), ctypes.c_void_p) for c in cargs]
        )
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        gx, gy, gz = (tuple(grid) + (1, 1))[:3]
        bx, by, bz = (tuple(block) + (1, 1))[:3]
        _check_cu(
            self._cuda,
            self._cuda.cuLaunchKernel(
                self._fn, gx, gy, gz, bx, by, bz, shmem, ctypes.c_void_p(stream), pointers, None
            ),
        )


class Module:
    """One compiled CUDA source; ``get(name)`` returns launchable kernels."""

    def __init__(self, source: str, *, name: str = "kernels.cu", defines: dict[str, object] | None = None,
                 device: torch.device | None = None) -> None:
        nvrtc, cuda = _libs()
        if device is not None and device.index is not None:
            torch.cuda.set_device(device.index)
        torch.cuda.init()
        torch.zeros(1, device=device or "cuda")  # make torch's primary context current
        major, minor = torch.cuda.get_device_capability(device)
        prog = ctypes.c_void_p()
        _check_nvrtc(
            nvrtc,
            nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None),
        )
        options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17", b"--use_fast_math"]
        for key, value in (defines or {}).items():
            options.append(f"-D{key}={value}".encode())
        c_options = (ctypes.c_char_p * len(options))(*options)
        _check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(prog, len(options), c_options), prog)
        size = ctypes.c_size_t()
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)))
        cubin = ctypes.create_string_buffer(size.value)
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetCUBIN(prog, cubin))
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        self._cuda = cuda
        self._module = ctypes.c_void_p()
        _check_cu(cuda, cuda.cuModuleLoadData(ctypes.byref(self._module), cubin))

    def get(self, name: str) -> Kernel:
        fn = ctypes.c_void_p()
        _check_cu(self._cuda, self._cuda.cuModuleGetFunction(ctypes.byref(fn), self._module, name.encode()))
        return Kernel(self._cuda, fn)
