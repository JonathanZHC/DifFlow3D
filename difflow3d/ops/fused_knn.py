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
#if SEG
          const int* __restrict__ qseg,        // [B, N] segment id per query
          const int* __restrict__ rseg,        // [B, M] segment id per reference
#endif
          int N, int M)
{
    __shared__ float tile[TILE * 3];
#if SEG
    __shared__ int tile_seg[TILE];
#endif
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * WARPS + warp;
    const bool active = n < N;
    float qx = 0.f, qy = 0.f, qz = 0.f;
#if SEG
    int qs = -1;
#endif
    if (active) {
        const float* q = query + ((size_t)b * N + n) * 3;
        qx = q[0]; qy = q[1]; qz = q[2];
#if SEG
        qs = qseg[(size_t)b * N + n];
#endif
    }
    float best_d[K];
    int   best_i[K];
#pragma unroll
    for (int k = 0; k < K; ++k) { best_d[k] = INF; best_i[k] = 0; }

    const float* ref = reference + (size_t)b * M * 3;
    int range_begin = 0, range_end = M;
#if SEG
    // Segments are contiguous and ascending in both inputs (object-major): the block only scans the references of
    // the segments its queries belong to, [lower_bound(first query's id), upper_bound(last query's id)).
    __shared__ int range[2];
    if (threadIdx.x == 0) {
        const int* rs = rseg + (size_t)b * M;
        const int first = qseg[(size_t)b * N + blockIdx.x * WARPS];
        const int last = qseg[(size_t)b * N + min(N - 1, blockIdx.x * WARPS + WARPS - 1)];
        int lo = 0, hi = M;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (rs[mid] < first) lo = mid + 1; else hi = mid; }
        range[0] = lo;
        hi = M;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (rs[mid] <= last) lo = mid + 1; else hi = mid; }
        range[1] = lo;
    }
    __syncthreads();
    range_begin = range[0]; range_end = range[1];
#endif
    for (int base = range_begin; base < range_end; base += TILE) {
        const int count = min(TILE, range_end - base);
        for (int i = threadIdx.x; i < count * 3; i += BLOCK) tile[i] = ref[(size_t)base * 3 + i];
#if SEG
        for (int i = threadIdx.x; i < count; i += BLOCK) tile_seg[i] = rseg[(size_t)b * M + base + i];
#endif
        __syncthreads();
        if (active) {
            for (int j = lane; j < count; j += 32) {
#if SEG
                if (tile_seg[j] != qs) continue;     // other object: never a neighbour
#endif
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
    int found = K;
    for (int r = 0; r < K; ++r) {
        float head = best_d[0];
        float m = head;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fminf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (m >= INF) { found = r; break; }      // segment smaller than K (uniform across the warp)
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
    // Fewer than K candidates in the query's segment: cycle through the ones found (every member of a small
    // object becomes a neighbour with near-equal multiplicity; never a point of another object).
    // No candidate at all (a padding segment's level-1 grouping, model.segments): index 0, a valid read whose
    // result is never used.
    if (lane == 0)
        for (int r = found; r < K; ++r) out[r] = found > 0 ? out[r % found] : 0;
}
"""

_kernels: dict[tuple[int, int, bool], object] = {}
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


def _kernel(k: int, device: torch.device, segmented: bool = False):
    global _disabled
    key = (int(k), device.index if device.index is not None else torch.cuda.current_device(), bool(segmented))
    kernel = _kernels.get(key)
    if kernel is None:
        try:
            kernel = Module(_SOURCE, name="fused_knn.cu", defines={"K": int(k), "SEG": int(bool(segmented))},
                            device=device).get("knn_exact")
        except Exception:  # pragma: no cover - depends on local CUDA toolchain
            _disabled = True
            raise
        _kernels[key] = kernel
    return kernel


def knn_indices(query: torch.Tensor, reference: torch.Tensor, k: int,
                query_seg: torch.Tensor | None = None, reference_seg: torch.Tensor | None = None) -> torch.Tensor:
    """``[B,N,3]`` queries, ``[B,M,3]`` references -> ``[B,N,k]`` int64 indices into M.

    With ``query_seg`` / ``reference_seg`` (``[B,N]`` / ``[B,M]`` int32 segment ids, each ascending: object-major
    inputs) only references of the query's segment are candidates and only that range is scanned; a segment with
    fewer than k references repeats its neighbours cyclically."""
    query = query.contiguous()
    reference = reference.contiguous()
    B, N, _ = query.shape
    M = int(reference.shape[1])
    out = torch.empty((B, N, int(k)), dtype=torch.int64, device=query.device)
    if N == 0:
        return out
    segmented = query_seg is not None
    kernel = _kernel(int(k), query.device, segmented)
    grid = ((N + WARPS - 1) // WARPS, B)
    if segmented:
        query_seg = query_seg.to(torch.int32).contiguous()
        reference_seg = reference_seg.to(torch.int32).contiguous()
        assert query_seg.shape == (B, N) and reference_seg.shape == (B, M)
        kernel.launch(grid, (BLOCK,), query, reference, out, query_seg, reference_seg, int(N), M)
    else:
        kernel.launch(grid, (BLOCK,), query, reference, out, int(N), M)
    return out
