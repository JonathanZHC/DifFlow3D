#!/usr/bin/env python3
"""The Triton furthest point sampling returns exactly the indices of the pointnet2 CUDA kernel.

Random clouds, exact ties (duplicated points, quantised grids), non-power-of-two sizes and model-space coordinates.
Inside the perceptive-safety-filter container:
    PYTHONPATH=/workspace/external/ScenePredictor/DifFlow3D /opt/tracking-venv/bin/python scripts/test_fps_triton.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from difflow3d.ops.pointnet2 import fps_triton  # noqa: E402
from difflow3d.ops.pointnet2 import pointnet2_utils  # noqa: E402


def cuda_reference(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    out = torch.empty((B, npoint), device=xyz.device, dtype=torch.int32)
    temp = torch.full((B, N), 1e10, device=xyz.device, dtype=torch.float32)
    pointnet2_utils.pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, out)
    return out


def main() -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    center = torch.tensor([-0.52, -1.57, 21.36], device=dev)
    cases = []
    for n, m in ((8192, 2048), (4096, 1024), (3000, 1024), (2048, 1024), (1024, 256), (1000, 256), (512, 128),
                 (256, 64), (100, 32)):
        for _ in range(4):
            cases.append((f"random n={n}", torch.randn(2, n, 3, device=dev) * 3 + 20, m))
        cases.append((f"model-space n={n}", (torch.randn(1, n, 3, device=dev) * 9 + center).contiguous(), m))
        base = torch.randn(1, max(n // 4, 1), 3, device=dev) * 3 + 20
        cases.append((f"duplicates n={n}", base.repeat(1, 4, 1)[:, :n].contiguous(), m))
        grid = torch.randint(0, 6, (1, n, 3), device=dev).float() * 0.5
        cases.append((f"grid ties n={n}", grid.contiguous(), m))
        # Near ties: distances equal up to the last bits, where the rounding of the distance (nvcc's FMA
        # contraction) decides the pick. Model-space scale like the deployed inputs.
        for _ in range(6):
            near = grid * 3.0 + center + torch.randn(1, n, 3, device=dev) * 2e-6
            cases.append((f"near ties n={n}", near.contiguous(), m))
    bad = [name for name, xyz, m in cases
           if not torch.equal(cuda_reference(xyz, m), fps_triton.furthest_point_sample(xyz, m))]
    assert not bad, f"index mismatch in {len(bad)} of {len(cases)} cases: {bad[:5]}"
    print(f"[ok] {len(cases)} cases bit-identical to the CUDA kernel")
    # The public entry point uses Triton by default.
    xyz = torch.randn(1, 4096, 3, device=dev) + 20
    assert torch.equal(pointnet2_utils.furthest_point_sample(xyz, 1024), cuda_reference(xyz, 1024))
    # Inside a torch.compile region (difflow.runtime.compile_mode compiles PointConvD.forward, which samples here).
    compiled = torch.compile(lambda points: pointnet2_utils.furthest_point_sample(points, 1024), dynamic=False)
    for _ in range(3):
        xyz = (torch.randn(1, 4096, 3, device=dev) * 9 + center).contiguous()
        assert torch.equal(compiled(xyz), cuda_reference(xyz, 1024)), "compiled FPS differs from the CUDA kernel"
    print("[ok] identical inside torch.compile")

    for n, m in ((4096, 1024), (1024, 256)):
        xyz = (torch.randn(1, n, 3, device=dev) * 3 + 20).contiguous()
        times = {}
        for name, fn in (("cuda", cuda_reference), ("triton", fps_triton.furthest_point_sample)):
            for _ in range(5):
                fn(xyz, m)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn(xyz, m)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                graph.replay()
            torch.cuda.synchronize()
            times[name] = (time.perf_counter() - t0) * 1e3 / 50
        print(f"[time] n={n} m={m}: cuda {times['cuda']:.3f} ms, triton {times['triton']:.3f} ms (CUDA graph replay)")
    print("ALL FPS TESTS PASSED")


if __name__ == "__main__":
    main()
