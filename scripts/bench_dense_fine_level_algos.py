# Exploratory (2026-10-02): fork-model monkeypatches evaluated inside the perceptive-safety-filter container (/tmp/real = ~/datasets/scene_flow/real). Not wired into the pipeline.
"""Algorithmic speed-ups of the fine level at 8192 points (fork model, recorded human pseudo GT, eager timing).
Base  : 8192, 4/2/2, top-level FPS -> strided (Morton-sorted input).
H     : + hierarchical cosine kNN at the fine level (candidates = 8 children of the 16 level-1 matches of the parent).
M     : + two-resolution fine level: 1 iteration at 4096 (strided subset) then 1 at 8192 (shared weights).
MH    : M with hierarchical cosine at both stages.   MS: M with hierarchical at 4096 and spatial-32 candidates at 8192."""
import sys, glob, time, os, json
sys.path.insert(0, "/opt/DifFlow3D"); sys.path.insert(0, "/tmp")
import numpy as np, torch
import fast8192 as F
from difflow3d.model import recurrent as R, pointconv as PC
from difflow3d.model.encoder import _self_knn_context
import difflow3d.ops.pointnet2.pointnet2_utils as PU

TOP = 2048
G = {}                                                   # level-1 context recorded while recurrent1 runs
ORIG_FWD, ORIG_COS = F.ORIG_FWD, F.ORIG_COS
MODEL = [None]
CFG = {"hier_fine": False, "two_res": False, "fine_mode": "hier"}   # fine_mode for the 8192 stage of two_res: hier | spatial | full


def cos_full(self, feat1, feat2):
    return ORIG_COS(self, feat1, feat2)


def hier_candidates(pc1, pc2, pc1_l1, pc2_l1, cos12_l1, cos21_l1, children=8):
    p1, p2 = pc1.permute(0, 2, 1).contiguous(), pc2.permute(0, 2, 1).contiguous()
    q1, q2 = pc1_l1.permute(0, 2, 1).contiguous(), pc2_l1.permute(0, 2, 1).contiguous()
    par1 = PC.knn_point(1, q1, p1)[..., 0]; par2 = PC.knn_point(1, q2, p2)[..., 0]           # [B,N0]
    ch2 = PC.knn_point(children, p2, q2); ch1 = PC.knn_point(children, p1, q1)                   # [B,N1,8]
    B = p1.shape[0]
    m12 = torch.gather(cos12_l1, 1, par1[..., None].expand(-1, -1, cos12_l1.shape[2]))            # [B,N0,16] level-1 matches of the parent
    c12 = torch.gather(ch2[:, :, None, :].expand(-1, -1, m12.shape[2], -1), 1, m12[..., None].expand(-1, -1, -1, children)).reshape(B, p1.shape[1], -1)
    m21 = torch.gather(cos21_l1, 1, par2[..., None].expand(-1, -1, cos21_l1.shape[2]))
    c21 = torch.gather(ch1[:, :, None, :].expand(-1, -1, m21.shape[2], -1), 1, m21[..., None].expand(-1, -1, -1, children)).reshape(B, p2.shape[1], -1)
    return c12, c21


def cos_from_candidates(self, feat1, feat2, c12, c21):
    half = self.flow_nei // 2
    f1n = PC.l2_normalize_points(feat1.permute(0, 2, 1)); f2n = PC.l2_normalize_points(feat2.permute(0, 2, 1))
    s12 = (PC.index_points_group(f2n, c12) * f1n.unsqueeze(2)).sum(-1); s21 = (PC.index_points_group(f1n, c21) * f2n.unsqueeze(2)).sum(-1)
    return torch.gather(c12, 2, s12.topk(half, -1, largest=True, sorted=False).indices), torch.gather(c21, 2, s21.topk(half, -1, largest=True, sorted=False).indices)


def cos_dispatch(self, feat1, feat2):
    mode = getattr(self, "_cos_mode", "full")
    if mode == "full":
        return ORIG_COS(self, feat1, feat2)
    if mode == "hier":
        c12, c21 = hier_candidates(*self._pc, G["pc1"], G["pc2"], G["cos12"], G["cos21"])
    else:                                                                                       # spatial candidates around the warped target
        p1 = self._pc[0].permute(0, 2, 1).contiguous(); p2w = self._pc2w.permute(0, 2, 1).contiguous()
        c12, c21 = PC.knn_point(32, p2w, p1), PC.knn_point(32, p1, p2w)
    return cos_from_candidates(self, feat1, feat2, c12, c21)


def fwd(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, gt_flow=None, certainty=None, uncertainty=0.5, self_knn_context=None):
    N = pc1.shape[2]
    if N <= TOP:                                                                               # coarse / middle levels unchanged; record level-1 context
        self._cos_mode = "full"
        out = ORIG_FWD(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, gt_flow, certainty, uncertainty, self_knn_context)
        if N == 1024:
            G["pc1"], G["pc2"] = pc1, pc2
            G["cos12"], G["cos21"] = ORIG_COS(self, feat1, feat2)
        return out
    self._pc = (pc1, pc2)
    if not CFG["two_res"]:
        self._cos_mode = "hier" if CFG["hier_fine"] else "full"
        self._pc2w = self.warping(pc1, pc2, up_flow)
        return ORIG_FWD(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, gt_flow, certainty, uncertainty, self_knn_context)
    # ---- two-resolution fine level ----
    m = MODEL[0]; iters = self.iters; self.iters = 1
    S = torch.arange(0, N, 2, device=pc1.device)                                              # Morton-sorted input -> strided = uniform
    sub = lambda t: t.index_select(2, S)
    pc1m, pc2m = sub(pc1), sub(pc2)
    self._pc = (pc1m, pc2m); self._cos_mode = "hier" if CFG["fine_mode"] != "full" else "full"
    self._pc2w = self.warping(pc1m, pc2m, sub(up_flow))
    knn_m = _self_knn_context(pc1m, int(self.flow.nsample))
    flows_m, f1n_m, f2n_m, ff_m, cert_m = ORIG_FWD(self, pc1m, pc2m, sub(feat1_new), sub(feat2_new), sub(feat1), sub(feat2), sub(up_flow), sub(up_feat), None, sub(certainty) if certainty is not None else None, uncertainty, knn_m)
    ctx1 = m._prepare_upsample_context(pc1, pc1m); ctx2 = m._prepare_upsample_context(pc2, pc2m)
    up = lambda t, c: m._apply_upsample_context(t, c)
    self._pc = (pc1, pc2); self._cos_mode = CFG["fine_mode"]
    self._pc2w = self.warping(pc1, pc2, up(flows_m[-1], ctx1))
    flows_f, f1n, f2n, ff, cert = ORIG_FWD(self, pc1, pc2, up(f1n_m, ctx1), up(f2n_m, ctx2), feat1, feat2, up(flows_m[-1], ctx1), up(ff_m, ctx1), None, up(cert_m, ctx1) if cert_m is not None else None, uncertainty, self_knn_context)
    self.iters = iters
    return ([up(flows_m[-1], ctx1)] + list(flows_f), f1n, f2n, ff, cert)


R.RecurrentUnit._prepare_cosine_neighbors = cos_dispatch
R.RecurrentUnit.forward = fwd
PU.furthest_point_sample = F.rand_subset                                                       # top-level strided (TOP=2048 in fast8192)
F.MODE = "strided"

VARIANTS = [("Base", dict(hier_fine=False, two_res=False, fine_mode="full")), ("H hier", dict(hier_fine=True, two_res=False, fine_mode="full")),
            ("M 4096+8192", dict(hier_fine=False, two_res=True, fine_mode="full")), ("MH", dict(hier_fine=False, two_res=True, fine_mode="hier")),
            ("MS", dict(hier_fine=False, two_res=True, fine_mode="spatial"))]
NPTS = int(os.environ.get("NPTS", 8192)); SEL = os.environ.get("VARS", "")
if SEL: VARIANTS = [v for v in VARIANTS if v[0].split()[0] in SEL.split(",")]
BINS = [0, 0.002, 0.01, 0.02, 0.05, 1.0]
out = {}
if __name__ == "__main__":
    for name, cfg in VARIANTS:
        CFG.update(cfg); m = F.build((4, 2, 2)); MODEL[0] = m; rng = np.random.default_rng(0); F._cache.clear()
        err, gn, times = [], [], []
        with torch.inference_mode():
            for scene in ("S1_static", "S2_human", "S3_two_sides"):
                fs = sorted(glob.glob(f"/tmp/real/{scene}/gap1/pair_*.npz")); z0 = np.load(fs[0]); ext = z0["p1"].max(0) - z0["p1"].min(0); s = (5.0 / float(np.prod(ext))) ** (1 / 3)
                for f in fs:
                    z, Gt = np.load(f), np.load(f.replace("pair_", "gt_"))
                    i1 = F.sample(z["p1"].shape[0], NPTS, rng, z["p1"]); i2 = F.sample(z["p2"].shape[0], NPTS, rng, z["p2"])
                    a = torch.from_numpy(z["p1"][i1] * s).cuda()[None]; b = torch.from_numpy(z["p2"][i2] * s).cuda()[None]
                    torch.cuda.synchronize(); t0 = time.time()
                    ea = m.encode_frame(a, a); eb = m.encode_frame(b, b); o = m.decode_pair(ea, eb, None, 0.1)
                    torch.cuda.synchronize(); times.append(time.time() - t0)
                    fw = o[0][0][0].permute(0, 2, 1)[0].float().cpu().numpy() / s
                    k = Gt["matched"][i1]; err.append(np.linalg.norm(fw - Gt["disp"][i1], axis=1)[k]); gn.append(np.linalg.norm(Gt["disp"][i1], axis=1)[k])
        e, g = np.concatenate(err), np.concatenate(gn); bb = np.digitize(g, BINS) - 1
        r = {"ms_eager_2enc": float(np.median(times) * 1e3), "epe": float(e.mean() * 1e3), "bins": [float(e[bb == i].mean() * 1e3) for i in range(len(BINS) - 1)]}
        out[name] = r
        print(f"{name:12s} eager {r['ms_eager_2enc']:5.1f} ms | EPE {r['epe']:.2f} mm | static<2 {r['bins'][0]:.2f} | 2-10 {r['bins'][1]:.2f} | 10-20 {r['bins'][2]:.2f} | 20-50 {r['bins'][3]:.2f} | >50 {r['bins'][4]:.2f}", flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(out, open(f"/tmp/algo{NPTS}.json", "w"), indent=1)
