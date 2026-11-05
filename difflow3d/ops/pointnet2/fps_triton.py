"""Exact furthest point sampling in Triton, bit-identical to pointnet2's furthest_point_sampling_kernel.

One program per batch element keeps all points in registers; each of the m-1 iterations updates the running
minimum distance and picks the next point with ONE block-wide max reduction over a packed int64 key:
(float bits of the distance << 32) | (INT32_MAX - tie_key), tie_key = bitrev(k mod BS) * 2^16 + k with
BS = min(2^floor(log2 n), 1024) the CUDA kernel's block size. Distances are non-negative, so their float bits order
like the floats. Ties: within a thread (k = tid + i*BS) the CUDA kernel keeps the lowest k; its shared-memory tree
(slot s vs s+h, h = BS/2 .. 1, left kept on ties) prefers threads whose low bits are zero, bit 0 decided last, i.e.
the smallest bit-reversed tid. The distance is fma(dz, dz, fma(dx, dx, dy * dy)): nvcc contracts
dx*dx + dy*dy + dz*dz by fusing each product on the left into the following add. Identified by running every
contraction order on real near-tie inputs; Triton's own FP fusion is disabled so nothing else is contracted.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_INT32_MAX = 2147483647


@triton.jit
def _fps_kernel(xyz_ptr, out_ptr, n, m, BS: tl.constexpr, LOG_BS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    valid = offs < n
    base = xyz_ptr + b.to(tl.int64) * n * 3
    x = tl.load(base + offs * 3 + 0, mask=valid, other=0.0)
    y = tl.load(base + offs * 3 + 1, mask=valid, other=0.0)
    z = tl.load(base + offs * 3 + 2, mask=valid, other=0.0)
    dist = tl.full([BLOCK], 1e10, tl.float32)
    tid = offs % BS
    rev = tl.zeros([BLOCK], tl.int32)
    for i in tl.static_range(LOG_BS):
        rev = rev | (((tid >> i) & 1) << (LOG_BS - 1 - i))
    tie = (2147483647 - (rev * 65536 + offs)).to(tl.int64)
    out = out_ptr + b.to(tl.int64) * m
    tl.store(out, 0)
    old = 0
    for j in range(1, m):
        x1 = tl.load(base + old * 3 + 0)
        y1 = tl.load(base + old * 3 + 1)
        z1 = tl.load(base + old * 3 + 2)
        dx = x - x1
        dy = y - y1
        dz = z - z1
        d = tl.fma(dz, dz, tl.fma(dx, dx, dy * dy))
        dist = tl.minimum(d, dist)
        bits = dist.to(tl.int32, bitcast=True).to(tl.int64)
        packed = tl.where(valid, (bits << 32) | tie, -1)
        best = tl.max(packed, axis=0)
        old = ((2147483647 - (best & 0xFFFFFFFF)) % 65536).to(tl.int32)
        tl.store(out + j, old)


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """(B, N, 3) float32 contiguous CUDA -> (B, npoint) int32, identical to pointnet2 furthest_point_sample."""
    assert xyz.is_cuda and xyz.is_contiguous() and xyz.dtype == torch.float32 and xyz.shape[-1] == 3
    B, N, _ = xyz.shape
    if not 1 <= N <= 65536:
        raise ValueError(f"fps_triton needs 1..65536 points, got {N}")
    out = torch.empty((B, int(npoint)), device=xyz.device, dtype=torch.int32)
    if npoint <= 0:
        return out
    block_size = max(min(1 << int(math.log(N) / math.log(2.0)), 1024), 1)   # opt_n_threads(N)
    block = triton.next_power_of_2(N)
    # Warp counts tuned once on an RTX 5090 (CUDA-graph replay): 4 up to 4096 points, 8 at 8192, 16 above (untuned).
    num_warps = max(1, min(4, block // 32)) if block <= 4096 else (8 if block <= 8192 else 16)
    _fps_kernel[(B,)](xyz, out, N, int(npoint), BS=block_size, LOG_BS=block_size.bit_length() - 1, BLOCK=block,
                      num_warps=num_warps, enable_fp_fusion=False)
    return out
