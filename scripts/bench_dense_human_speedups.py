# Exploratory benchmark (2026-10-02): runs inside the perceptive-safety-filter container with /tmp/real = ~/datasets/scene_flow/real (human pairs + pseudo GT). Monkeypatches the model; not wired into the pipeline.
"""Fork model on the recorded human (pseudo GT): accuracy and eager time of the 8192-point speed-ups.
A: 8192 4/2/2 as is | B: + random fixed subset instead of FPS | C: + cosine kNN restricted to 64 spatial candidates
D: + fine iterations 1 | E: + fp16 autocast. Reference: 1024 4/2/2."""
import sys, glob, time, os, json
sys.path.insert(0, "/opt/DifFlow3D")
import numpy as np, torch
from difflow3d.model.difflow import PointConvBidirection
from difflow3d.model import recurrent as R, pointconv as PC
import difflow3d.ops.pointnet2.pointnet2_utils as PU
from difflow3d.runtime.checkpoint import load_checkpoint
from difflow3d.runtime.inference import configure_fast_inference
configure_fast_inference(True)

ORIG_FPS, ORIG_COS, ORIG_FWD = PU.furthest_point_sample, R.RecurrentUnit._prepare_cosine_neighbors, R.RecurrentUnit.forward
_cache = {}
TOP = int(os.environ.get("TOP", 2048))                     # replace FPS only when N > TOP (the expensive top level)
def rand_subset(xyz, npoint):
    B, N, _ = xyz.shape
    if N <= TOP:
        return ORIG_FPS(xyz, npoint)
    k = (N, npoint)
    if k not in _cache:
        if MODE == "strided":                                   # input is Morton-sorted -> every (N/npoint)-th point is spatially uniform
            _cache[k] = (torch.arange(npoint) * (N // npoint)).int().cuda()[None].expand(B, -1).contiguous()
        else:
            _cache[k] = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:npoint].sort().values.int().cuda()[None].expand(B, -1).contiguous()
    return _cache[k]
MODE = os.environ.get("MODE", "strided")
def morton_order(p):
    q = ((p - p.min(0)) / 0.01).astype(np.int64)
    def spread(v):
        v = v & 0x1FFFFF; v = (v | v << 32) & 0x1F00000000FFFF; v = (v | v << 16) & 0x1F0000FF0000FF; v = (v | v << 8) & 0x100F00F00F00F00F
        v = (v | v << 4) & 0x10C30C30C30C30C3; v = (v | v << 2) & 0x1249249249249249; return v
    return np.argsort(spread(q[:, 0]) | spread(q[:, 1]) << 1 | spread(q[:, 2]) << 2)
CAND = int(os.environ.get("CAND", 32))
def restricted_cos(self, feat1, feat2):
    half = self.flow_nei // 2
    f1n = PC.l2_normalize_points(feat1.permute(0, 2, 1)); f2n = PC.l2_normalize_points(feat2.permute(0, 2, 1))
    c12, c21 = self._cand
    s12 = (PC.index_points_group(f2n, c12) * f1n.unsqueeze(2)).sum(-1); s21 = (PC.index_points_group(f1n, c21) * f2n.unsqueeze(2)).sum(-1)
    return torch.gather(c12, 2, s12.topk(half, -1, largest=True, sorted=False).indices), torch.gather(c21, 2, s21.topk(half, -1, largest=True, sorted=False).indices)
def fwd_with_cand(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, *a, **kw):
    w = self.warping(pc1, pc2, up_flow)
    p1, p2 = pc1.permute(0, 2, 1), w.permute(0, 2, 1)
    self._cand = (PC.knn_point(CAND, p2, p1), PC.knn_point(CAND, p1, p2))
    return ORIG_FWD(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, *a, **kw)
def apply(flags):
    PU.furthest_point_sample = rand_subset if "rand" in flags else ORIG_FPS
    R.RecurrentUnit._prepare_cosine_neighbors = restricted_cos if "cos" in flags else ORIG_COS
    R.RecurrentUnit.forward = fwd_with_cand if "cos" in flags else ORIG_FWD

def build(it):
    m = PointConvBidirection(iters=4, coarse_iters=it[0], middle_iters=it[1], fine_iters=it[2])
    load_checkpoint(m, "/opt/DifFlow3D/checkpoints/model_difflow_355_0.0114.pth", strict=True); m.cuda().eval()
    for l in m.modules():
        if isinstance(l, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)): l.track_running_stats = False
    return m

def sample(n, N, rng, pts=None):
    idx = rng.choice(n, N, replace=False) if N <= n else np.concatenate([np.arange(n), rng.choice(n, N - n, replace=True)])
    return idx[morton_order(pts[idx])] if pts is not None else idx

CONFIGS = [("A 8192 4/2/2", 8192, (4, 2, 2), ()), ("B top-level strided", 8192, (4, 2, 2), ("rand",)),
           ("C +cos32", 8192, (4, 2, 2), ("rand", "cos")), ("D +fine1", 8192, (4, 2, 1), ("rand", "cos")), ("E +fp16", 8192, (4, 2, 1), ("rand", "cos", "fp16")),
           ("F B+fp16 (4/2/2)", 8192, (4, 2, 2), ("rand", "fp16")), ("G 4096 B+fp16", 4096, (4, 2, 2), ("rand", "fp16"))]
BINS = [0, 0.002, 0.01, 0.02, 0.05, 1.0]
out = {}
for name, N, it, flags in (CONFIGS if __name__ == "__main__" else []):
    apply(flags); m = build(it); rng = np.random.default_rng(0)
    err, gn, times = [], [], []
    ac = torch.autocast("cuda", dtype=torch.float16) if "fp16" in flags else torch.autocast("cuda", enabled=False)
    with torch.inference_mode(), ac:
        for scene in ("S1_static", "S2_human", "S3_two_sides"):
            fs = sorted(glob.glob(f"/tmp/real/{scene}/gap1/pair_*.npz")); z0 = np.load(fs[0]); ext = z0["p1"].max(0) - z0["p1"].min(0); s = (5.0 / float(np.prod(ext))) ** (1 / 3)
            for f in fs:
                z, G = np.load(f), np.load(f.replace("pair_", "gt_"))
                i1 = sample(z["p1"].shape[0], N, rng, z["p1"]); i2 = sample(z["p2"].shape[0], N, rng, z["p2"])
                a = torch.from_numpy(z["p1"][i1] * s).cuda()[None]; b = torch.from_numpy(z["p2"][i2] * s).cuda()[None]
                torch.cuda.synchronize(); t0 = time.time()
                ea = m.encode_frame(a, a); eb = m.encode_frame(b, b); o = m.decode_pair(ea, eb, None, 0.1)
                torch.cuda.synchronize(); times.append(time.time() - t0)
                fw = o[0][0][0].permute(0, 2, 1)[0].float().cpu().numpy() / s
                k = G["matched"][i1]; err.append(np.linalg.norm(fw - G["disp"][i1], axis=1)[k]); gn.append(np.linalg.norm(G["disp"][i1], axis=1)[k])
    e, g = np.concatenate(err), np.concatenate(gn); bb = np.digitize(g, BINS) - 1
    r = {"ms_eager_2enc": float(np.median(times) * 1e3), "epe": float(e.mean() * 1e3), "bins": [float(e[bb == i].mean() * 1e3) for i in range(len(BINS) - 1)]}
    out[name] = r
    print(f"{name:16s} eager {r['ms_eager_2enc']:5.1f} ms | EPE {r['epe']:.2f} mm | static<2 {r['bins'][0]:.2f} | 2-10 {r['bins'][1]:.2f} | 10-20 {r['bins'][2]:.2f} | 20-50 {r['bins'][3]:.2f} | >50 {r['bins'][4]:.2f}", flush=True)
    del m; torch.cuda.empty_cache()
if __name__ == "__main__": json.dump(out, open("/tmp/fast8192.json", "w"), indent=1)
