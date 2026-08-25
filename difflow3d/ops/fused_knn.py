"""Exact fused K-nearest-neighbour search for small 3-D point sets (CUDA, NVRTC-JIT).

Replaces ``square_distance`` (K=3 GEMM writing a [N,M] fp32 matrix, ``-2*``, two
broadcast adds) + ``torch.topk`` — ~30 MB of traffic and 7 launches per 1024-point
query — with one kernel: each thread owns a query point, streams the reference
points through shared memory and keeps its K best in registers.

Distances are the exact fp32 ``sum((q-r)^2)``; the old path computed
``|q|^2+|r|^2-2qr`` in TF32, so neighbour sets differ only at (near-)ties.
Indices are returned ascending by distance as int64 (same dtype as ``topk``).
"""
from __future__ import annotations

import torch

from .nvrtc_jit import Module

MAX_K = 32
WARPS = 4
BLOCK = 32 * WARPS
TILE = 256

_SOURCE = r"""
#ifndef K
#define K 16
#endif
#define WARPS 4
#define BLOCK (32 * WARPS)
#define TILE 256
#define INF 3.0e38f

// One warp per query point. Lane l scans reference points l, l+32, ... keeping
// its own sorted top-K in registers (constant-index compare/swap chain, so it
// stays in registers). The 32 sorted lists are then merged with a K-round
// warp-min: each round the lanes' current heads are reduced with shuffles, the
// winning lane pops its head, and lane 0 writes the result.
extern "C" __global__ void __launch_bounds__(BLOCK)
knn_exact(const float* __restrict__ query,     // [B, N, 3]
          const float* __restrict__ reference, // [B, M, 3]
          long long* __restrict__ indices,     // [B, N, K]
          int N, int M)
{
    __shared__ float tile[TILE * 3];
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * WARPS + warp;
    const bool active = n < N;
    float qx = 0.f, qy = 0.f, qz = 0.f;
    if (active) {
        const float* q = query + ((size_t)b * N + n) * 3;
        qx = q[0]; qy = q[1]; qz = q[2];
    }
    float best_d[K];
    int   best_i[K];
#pragma unroll
    for (int k = 0; k < K; ++k) { best_d[k] = INF; best_i[k] = 0; }

    const float* ref = reference + (size_t)b * M * 3;
    for (int base = 0; base < M; base += TILE) {
        const int count = min(TILE, M - base);
        for (int i = threadIdx.x; i < count * 3; i += BLOCK) tile[i] = ref[(size_t)base * 3 + i];
        __syncthreads();
        if (active) {
            for (int j = lane; j < count; j += 32) {
                const float dx = qx - tile[j * 3 + 0];
                const float dy = qy - tile[j * 3 + 1];
                const float dz = qz - tile[j * 3 + 2];
                float cd = dx * dx + dy * dy + dz * dz;
                int ci = base + j;
                if (cd < best_d[K - 1]) {
#pragma unroll
                    for (int k = 0; k < K; ++k) {
                        if (cd < best_d[k]) {
                            const float td = best_d[k]; best_d[k] = cd; cd = td;
                            const int   ti = best_i[k]; best_i[k] = ci; ci = ti;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!active) return;

    // Warp merge of 32 sorted lists -> global top-K.
    long long* out = indices + ((size_t)b * N + n) * K;
    for (int r = 0; r < K; ++r) {
        float head = best_d[0];
        float m = head;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fminf(m, __shfl_xor_sync(0xffffffffu, m, off));
        // lowest lane holding the minimum wins (deterministic tie-break)
        const unsigned ballot = __ballot_sync(0xffffffffu, head == m);
        const int winner = __ffs(ballot) - 1;
        const int win_index = __shfl_sync(0xffffffffu, best_i[0], winner);
        if (lane == 0) out[r] = (long long)win_index;
        if (lane == winner) {
#pragma unroll
            for (int k = 0; k < K - 1; ++k) { best_d[k] = best_d[k + 1]; best_i[k] = best_i[k + 1]; }
            best_d[K - 1] = INF;
        }
    }
}
"""

_kernels: dict[tuple[int, int], object] = {}
_disabled = False


def available(query: torch.Tensor, reference: torch.Tensor, k: int) -> bool:
    return (
        not _disabled
        and query.is_cuda
        and reference.is_cuda
        and query.dtype == torch.float32
        and reference.dtype == torch.float32
        and query.dim() == 3
        and reference.dim() == 3
        and query.shape[2] == 3
        and reference.shape[2] == 3
        and query.shape[0] == reference.shape[0]
        and 1 <= int(k) <= MAX_K
        and int(reference.shape[1]) >= int(k)
    )


def _kernel(k: int, device: torch.device):
    global _disabled
    key = (int(k), device.index if device.index is not None else torch.cuda.current_device())
    kernel = _kernels.get(key)
    if kernel is None:
        try:
            kernel = Module(_SOURCE, name="fused_knn.cu", defines={"K": int(k)}, device=device).get("knn_exact")
        except Exception:  # pragma: no cover - depends on local CUDA toolchain
            _disabled = True
            raise
        _kernels[key] = kernel
    return kernel


def knn_indices(query: torch.Tensor, reference: torch.Tensor, k: int) -> torch.Tensor:
    """``[B,N,3]`` queries, ``[B,M,3]`` references -> ``[B,N,k]`` int64 indices into M."""
    query = query.contiguous()
    reference = reference.contiguous()
    B, N, _ = query.shape
    M = int(reference.shape[1])
    out = torch.empty((B, N, int(k)), dtype=torch.int64, device=query.device)
    if N == 0:
        return out
    kernel = _kernel(int(k), query.device)
    kernel.launch(((N + WARPS - 1) // WARPS, B), (BLOCK,), query, reference, out, int(N), M)
    return out
