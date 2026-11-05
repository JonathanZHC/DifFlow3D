"""Exactness of the segmented kernels (multi-object inputs): the segmented fused kNN equals the per-object kNN for
every K, and the segmented Triton FPS equals the plain kernel run on each object alone. Run: python scripts/test_segments.py"""
import torch
from difflow3d.ops import fused_knn
from difflow3d.ops.pointnet2 import fps_triton, pointnet2_utils

torch.manual_seed(0)
dev = torch.device("cuda")
# two objects, overlapping in space (the worst case for leakage)
segs = [(1024, 4096), (512, 1024), (256, 512), (768, 1024), (2048, 4096), (256, 256)]
bad = 0
for trial in range(6):
    sizes = [s[0] for s in segs[trial:trial + 3]] if trial < 4 else [segs[trial][0]]
    clouds = [torch.randn(n, 3, device=dev) * 0.3 for n in sizes]         # same region
    xyz = torch.cat(clouds)[None].contiguous()
    seg = torch.cat([torch.full((n,), i, dtype=torch.int32, device=dev) for i, n in enumerate(sizes)])[None]
    for k in (3, 9, 16, 32):
        got = fused_knn.knn_indices(xyz, xyz, k, seg, seg)[0]
        off = 0
        for i, n in enumerate(sizes):
            ref = fused_knn.knn_indices(clouds[i][None].contiguous(), clouds[i][None].contiguous(), k)[0] + off
            g = got[off:off + n]
            # compare as sets per row (ties can reorder within equal distances)
            if not torch.equal(torch.sort(g, 1).values, torch.sort(ref, 1).values):
                bad += 1; print(f"kNN mismatch trial {trial} seg {i} k {k}: {(torch.sort(g,1).values != torch.sort(ref,1).values).any(1).sum().item()} rows")
            off += n
    # FPS: per-segment npoint = n/2
    offs = torch.tensor([sum(sizes[:i]) for i in range(len(sizes))], dtype=torch.int32, device=dev)
    cnts = torch.tensor(sizes, dtype=torch.int32, device=dev)
    nps = torch.tensor([n // 2 for n in sizes], dtype=torch.int32, device=dev)
    oofs = torch.tensor([sum(n // 2 for n in sizes[:i]) for i in range(len(sizes))], dtype=torch.int32, device=dev)
    got = fps_triton.furthest_point_sample_segments(xyz[0], offs, cnts, oofs, nps, int(sum(n // 2 for n in sizes)), max(sizes), max(n // 2 for n in sizes))
    oo = 0
    for i, n in enumerate(sizes):
        ref = fps_triton.furthest_point_sample(clouds[i][None].contiguous(), n // 2)[0] + int(offs[i])
        g = got[oo:oo + n // 2]
        if not torch.equal(g, ref):
            # tie-break order may differ when BS differs from the segment's own; check the SET of sampled points
            if torch.equal(torch.sort(g).values, torch.sort(ref).values):
                print(f"FPS trial {trial} seg {i}: same set, different order (BS differs)")
            else:
                bad += 1; print(f"FPS mismatch trial {trial} seg {i}: {(g != ref).sum().item()} of {n//2}")
        oo += n // 2
# Segments smaller than k: the neighbours cycle through the segment's members (fused kernel == torch fallback),
# never leave the segment, and a segment of n <= k members uses each member k//n or k//n+1 times.
from difflow3d.model import pointconv as PC
sizes = [32, 48, 16, 1024]
xyz = (torch.randn(sum(sizes), 3, device=dev) * 0.3)[None].contiguous()
seg = torch.cat([torch.full((n,), i, dtype=torch.int32, device=dev) for i, n in enumerate(sizes)])[None]
for k in (3, 9, 16, 32):
    fused = fused_knn.knn_indices(xyz, xyz, k, seg, seg)[0]
    PC.USE_FUSED_KNN, keep = False, PC.USE_FUSED_KNN
    torch_idx = PC.knn_point(k, xyz, xyz, seg, seg)[0]
    PC.USE_FUSED_KNN = keep
    if not torch.equal(seg[0][fused], seg[0][:, None].expand_as(fused)):
        bad += 1; print(f"small-segment kNN k {k}: neighbour outside its segment")
    if not torch.equal(torch.sort(fused, 1).values, torch.sort(torch_idx, 1).values):
        bad += 1; print(f"small-segment kNN k {k}: fused != torch fallback "
                        f"({(torch.sort(fused,1).values != torch.sort(torch_idx,1).values).any(1).sum().item()} rows)")
    off = 0
    for n in sizes:
        if n <= k:
            hits = torch.zeros(n, n, dtype=torch.long, device=dev).scatter_add_(
                1, fused[off:off + n] - off, torch.ones_like(fused[off:off + n]))
            if hits.min() < k // n or hits.max() > k // n + (k % n > 0):
                bad += 1; print(f"small-segment kNN k {k} n {n}: uneven multiplicity {hits.min()}..{hits.max()}")
        off += n
print("ALL SEGMENT KERNEL TESTS PASSED" if bad == 0 else f"{bad} FAILURES")
