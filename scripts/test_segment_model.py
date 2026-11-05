"""Segment equivalence of the DifFlow3D model: several objects packed into one input with Segments give, per object,
the result of that object run alone (differences below each object's own chaotic floor). The diffusion noise is made a
deterministic function of the point coordinates so packed and solo runs share it.
Run: python scripts/test_segment_model.py [checkpoint] (default: the deployed ZU checkpoint of the umbrella repo)."""
import sys
import torch

from difflow3d.runtime.checkpoint import load_checkpoint  # noqa: E402
from difflow3d.runtime.runner import configure_fast_inference  # noqa: E402
from difflow3d.model import segments as S  # noqa: E402
from difflow3d.model.difflow import PointConvBidirection  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/difflow3d_zu/zu_p4-Z2u-zup_model_e09.pth"
configure_fast_inference(enable_tf32=False)        # fp32 matmuls: comparisons at 1e-4
dev = torch.device("cuda")
model = PointConvBidirection(iters=4, coarse_iters=4, middle_iters=2, fine_iters=2, level1_points=1024)
load_checkpoint(model, CKPT, strict=True)
model.configure_dense_points(fast_top_level_min_points=0, hier_cosine_min_points=0)   # plain paths first
model.eval().to(dev)
torch.set_grad_enabled(False)

# deterministic "noise": hash of the point coordinates (same point -> same noise, wherever it sits in the batch)
_real_randn_like = torch.randn_like
CUR = {}
def det_randn_like(t):
    xyz = CUR["xyz"]                              # [B,3,N] model units of the current unit's pc1
    key = (xyz * 1000.0).round().sum(1)           # [B,N]
    g = torch.sin(key * 12.9898 + 78.233) * 43758.5453
    base = (g - g.floor()) * 2.0 - 1.0            # [-1,1)
    return base.unsqueeze(1).expand(t.shape[0], t.shape[1], t.shape[2]).contiguous() * 0.5
torch.randn_like = det_randn_like
import difflow3d.model.recurrent as R
_orig_fwd = R.RecurrentUnit.forward
def fwd(self, pc1, *a, **k):
    CUR["xyz"] = pc1
    return _orig_fwd(self, pc1, *a, **k)
R.RecurrentUnit.forward = fwd

LAST = {}
model.recurrent0.register_forward_hook(lambda m, i, o: LAST.__setitem__("recurrent0", o[0][-1]))
torch.manual_seed(0)
C_REF = torch.tensor([-0.52, -1.57, 21.36], device=dev)
def obj(n, center, radius=8.97):
    p = torch.randn(n, 3, device=dev); p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-6) * radius * torch.rand(n, 1, device=dev) ** (1/3)
    return p + center

def run(p1, p2, seg=None, seg2=None):
    e1 = model.encode_frame(p1[None], segments=seg)
    e2 = model.encode_frame(p2[None], segments=seg if seg2 is None else seg2)
    flows = model.decode_pair(e1, e2, None, 0.1)[0]
    # The level-0 output is the LAST flow the fine recurrent unit produced (recurrent0.flows[-1]); take it from
    # the unit directly so a small object's identity level 1 (same N as level 0) cannot be mistaken for it.
    return LAST["recurrent0"]

bad = 0
MM_PER_UNIT = 1000.0 / 214.0     # synthetic objects: RMS radius 8.97 model units ~= 4.2 cm object -> k ~ 214 units per metre
def solo_seg(n):
    # objects below the 256-point single-cloud minimum run alone as a one-segment input
    return S.segments_for((n,), 1024, dev) if n < 256 else None


for sizes in ((1024, 512), (2048, 256), (4096, 768), (512, 256, 256), (1024, 1024), (1024, 32), (512, 64, 128, 48),
              (256, 32, 32)):
    objs1 = [obj(n, C_REF) for n in sizes]                     # all at the model centre (per-object normalisation)
    objs2 = [o + _real_randn_like(o) * 0.02 + torch.tensor([0.3, 0.0, 0.0], device=dev) for o in objs1]
    solo = [run(a, b, solo_seg(a.shape[0])) for a, b in zip(objs1, objs2)]
    # chaotic floor of each object alone: 1e-6 model units (~5 nm) of input jitter
    floor = [(run(a + 1e-6, b, solo_seg(a.shape[0])) - s).abs().max().item() for a, b, s in zip(objs1, objs2, solo)]
    seg = S.segments_for(sizes, 1024, dev)
    packed = run(torch.cat(objs1), torch.cat(objs2), seg)
    off = 0
    for i, n in enumerate(sizes):
        d = (packed[..., off:off + n] - solo[i]).abs().max().item()
        # rounding-level difference (the packed conv shapes differ): accept up to 3x this object's own
        # chaotic floor, and at most 0.5 mm
        tol = max(3.0 * floor[i], 1e-5)
        ok = d <= tol and d * MM_PER_UNIT < 0.5
        bad += 0 if ok else 1
        print(f"sizes {sizes} object {i} (n={n}): max |packed - solo| = {d:.2e} units = {d * MM_PER_UNIT:.3f} mm "
              f"(own jitter floor {floor[i]:.2e}) {'OK' if ok else 'MISMATCH'}")
        off += n

# Graph-shaped layouts: (a) a level-1 capacity above what the objects need adds a padding segment, which must not
# change any object; (b) the two frames of a pair split the bucket differently (one Segments per frame).
for sizes1, sizes2, capacity in (((2048, 256, 256), (2048, 256, 256), 2048), ((1536, 512), (1024, 1024), 2048),
                                 ((3760, 336), (3504, 592), 2048), ((1024, 32, 64), (992, 64, 64), 1120)):
    capacity = max(capacity, *(sum(S.level_sizes(n, 1024)[1] for n in sz) for sz in (sizes1, sizes2)))
    capacity = -(-capacity // 16) * 16
    objs1 = [obj(n, C_REF) for n in sizes1]
    objs2 = [obj(n, C_REF) + torch.tensor([0.3, 0.0, 0.0], device=dev) for n in sizes2]
    seg1 = S.Segments(sizes1, 1024, dev, level1_total=capacity)
    seg2 = S.Segments(sizes2, 1024, dev, level1_total=capacity)
    packed = run(torch.cat(objs1), torch.cat(objs2), seg1, seg2)
    off = 0
    for i, n in enumerate(sizes1):
        one1, one2 = S.Segments((n,), 1024, dev), S.Segments((sizes2[i],), 1024, dev)
        solo = run(objs1[i], objs2[i], one1, one2)
        floor = (run(objs1[i] + 1e-6, objs2[i], one1, one2) - solo).abs().max().item()
        d = (packed[..., off:off + n] - solo).abs().max().item()
        tol = max(3.0 * floor, 1e-5)
        ok = d <= tol and d * MM_PER_UNIT < 0.5
        bad += 0 if ok else 1
        print(f"layout {sizes1}->{sizes2} N1 capacity {capacity} (pad {seg1.totals[1] - sum(S.level_sizes(c, 1024)[1] for c in sizes1)}) "
              f"object {i}: max |packed - solo| = {d:.2e} units (floor {floor:.2e}) {'OK' if ok else 'MISMATCH'}")
        off += n
print("ALL SEGMENT MODEL TESTS PASSED" if bad == 0 else f"{bad} FAILURES")
