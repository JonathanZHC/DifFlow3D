"""Fused cross-block prologue for DifFlow3D (CUDA, NVRTC-JIT).

Computes, in one pass and one write of the ``[B,C,N,K]`` output::

    out[c,n,k] = act( points2[c, idx[n,k]] + points1[c,n] + sum_j Wpos[c,j]*dir[j,n,k] + bpos[c] )

which the eager path spelled as grouping (8.4 MB write) -> 1x1 conv on ``dir``
(8.4 MB write) -> two broadcast adds (~25 MB read, 8.4 MB write) -> leaky relu.
Roughly 50 MB of traffic per cross block becomes ~10 MB; there are 24 cross
blocks per decode replay.
"""
from __future__ import annotations

import torch

from .nvrtc_jit import Module

BLOCK = 256
CCHUNK = 32

_SOURCE = r"""
#define BLOCK 256
#define CCHUNK 32

extern "C" __global__ void __launch_bounds__(BLOCK)
cross_prologue(const float* __restrict__ points2,   // [B, C, M]
               const float* __restrict__ points1,   // [B, C, N]
               const int*   __restrict__ idx,       // [B, N, K]
               const float* __restrict__ dir,       // [B, 3, N, K]
               const float* __restrict__ wpos,      // [C, 3]
               const float* __restrict__ bpos,      // [C]
               float* __restrict__ out,             // [B, C, N, K]
               float slope, int C, int N, int K, int M)
{
    const int b = blockIdx.y;
    const int nk = blockIdx.x * BLOCK + threadIdx.x;    // n * K + k
    const int NK = N * K;
    if (nk >= NK) return;
    // Channels are split across blockIdx.z (CCHUNK each) so wide layers keep
    // enough threads in flight; each thread still reuses its gathered index and
    // direction across all of its channels.
    const int c_begin = blockIdx.z * CCHUNK;
    const int c_end = min(C, c_begin + CCHUNK);
    const int n = nk / K;
    const int j = idx[(size_t)b * NK + nk];
    const size_t dbase = (size_t)b * 3 * NK + nk;
    const float d0 = dir[dbase], d1 = dir[dbase + NK], d2 = dir[dbase + 2 * (size_t)NK];
    const float* p2 = points2 + (size_t)b * C * M + j;
    const float* p1 = points1 + (size_t)b * C * N + n;
    float* o = out + (size_t)b * C * NK + nk;
    for (int c = c_begin; c < c_end; ++c) {
        const float w0 = wpos[c * 3 + 0], w1 = wpos[c * 3 + 1], w2 = wpos[c * 3 + 2];
        float v = p2[(size_t)c * M] + p1[(size_t)c * N] + fmaf(w0, d0, fmaf(w1, d1, fmaf(w2, d2, bpos[c])));
        v = v > 0.f ? v : v * slope;
        o[(size_t)c * NK] = v;
    }
}
"""

_kernels: dict[int, object] = {}
_disabled = False


def _kernel(device: torch.device):
    global _disabled
    key = device.index if device.index is not None else torch.cuda.current_device()
    kernel = _kernels.get(key)
    if kernel is None:
        try:
            kernel = Module(_SOURCE, name="fused_cross_block.cu", device=device).get("cross_prologue")
        except Exception:  # pragma: no cover
            _disabled = True
            raise
        _kernels[key] = kernel
    return kernel


def supported(pos, bn, relu, points1, points2, idx_int, direction) -> bool:
    if _disabled or not points1.is_cuda:
        return False
    if not isinstance(pos, torch.nn.Conv2d) or tuple(pos.kernel_size) != (1, 1) or pos.in_channels != 3:
        return False
    if pos.bias is None or pos.groups != 1 or pos.stride != (1, 1) or pos.padding != (0, 0):
        return False
    if not isinstance(bn, torch.nn.Identity):
        return False
    if not isinstance(relu, (torch.nn.LeakyReLU, torch.nn.ReLU)):
        return False
    if points1.dtype != torch.float32 or points2.dtype != torch.float32 or direction.dtype != torch.float32:
        return False
    if idx_int.dtype != torch.int32:
        return False
    return points1.shape[1] == points2.shape[1] == pos.out_channels


def cross_prologue(pos, relu, points1, points2, idx_int, direction) -> torch.Tensor:
    """``[B,C,N,K]`` = act(group(points2) + points1 + pos(direction))."""
    points1 = points1.contiguous()
    points2 = points2.contiguous()
    direction = direction.contiguous()
    idx_int = idx_int.contiguous()
    B, C, N = points1.shape
    M = int(points2.shape[2])
    K = int(idx_int.shape[2])
    out = torch.empty((B, C, N, K), dtype=torch.float32, device=points1.device)
    if N * K == 0:
        return out
    weight = pos.weight.detach().reshape(C, 3)
    if not weight.is_contiguous():
        weight = weight.contiguous()
    bias = pos.bias.detach().contiguous()
    slope = float(relu.negative_slope) if isinstance(relu, torch.nn.LeakyReLU) else 0.0
    kernel = _kernel(points1.device)
    grid = ((N * K + BLOCK - 1) // BLOCK, B, (int(C) + CCHUNK - 1) // CCHUNK)
    kernel.launch(grid, (BLOCK,), points2, points1, idx_int, direction,
                  weight, bias, out, slope, int(C), int(N), K, M)
    return out
