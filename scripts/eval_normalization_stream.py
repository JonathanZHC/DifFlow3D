"""Streaming evaluation of the canonical normalisation in the runner (encoder reuse, buckets, CUDA graphs).
Full recordings in cycle order (tracked human, track id 1); a stream ends only when the human is absent (the
recorder skipped cycles, which are longer steps of the same stream). EPE on the cycles with a pseudo-GT pair (gap 1).

    /opt/tracking-venv/bin/python scripts/eval_normalization_stream.py --recordings <dir> --gt <dir> [--out x.json]
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_variable_points import build_model
from difflow3d.runtime.runner import DifFlow3DStreamingCudaGraphRunner
from difflow3d.runtime.normalization import AnchoredNormalization

REC, GT = "/workspace/results/eas_bench", "/tmp/real"
SCENES = ("S1_static", "S2_human", "S3_two_sides")
BUCKETS = (1024, 2048, 3072, 4096)
BINS = [0, 0.002, 0.01, 0.02, 0.05, 10.0]

def runner(model, norm=None):
    return DifFlow3DStreamingCudaGraphRunner(model, batch_size=1, num_points=4096, uncertainty=0.1, warmup=3, dt_s=1.0,
        second_base_voxel_size_m=0.01, second_candidate_ratio=1.1, auto_spatial_scale=norm is None, target_model_volume=5.0,
        fixed_spatial_scale=1.0, final_selection="uniform", point_buckets=BUCKETS, sort_anchors_morton=True,
        enable_profiling=True, normalization=norm)

VARIANTS = {
    "A deployed (5 m^3, first frame)": None,
    "B per frame + re-encode": dict(per_frame=True),
    "C per frame, no re-encode": dict(per_frame=True, restage_previous=False),
    "D hysteresis 1.5 / 0.3": dict(scale_ratio=1.5, center_shift=0.3),
    "E hysteresis 2.0 / 0.5": dict(scale_ratio=2.0, center_shift=0.5),
}

def load_frames(scene):
    out = []
    for f in sorted(glob.glob(f"{REC}/{scene}/cycle_*.npz")):
        d = np.load(f); cyc = int(os.path.basename(f)[6:13])
        h = d["points"][d["track_ids"] == 1].astype(np.float32)
        out.append((cyc, h))
    return out

def main():
    torch.manual_seed(0)
    model = build_model(4096, 4096)
    frames = {s: load_frames(s) for s in SCENES}
    gts = {}
    for s in SCENES:
        for f in glob.glob(f"{GT}/{s}/gap1/gt_*.npz"):
            cyc = int(os.path.basename(f)[3:10]); g = np.load(f); p = np.load(f.replace("gt_", "pair_"))
            gts[(s, cyc)] = (g["disp"], g["matched"], p["p1"].shape[0])
    res = {}
    for name, kw in VARIANTS.items():
        norm = AnchoredNormalization(**kw) if kw is not None else None
        r = runner(model, norm)
        err, gn, after, ms, wall, n_frames, resets, stream_len, cur = [], [], [], [], [], 0, 0, [], 0
        for s in SCENES:
            if norm is None:
                r.reset_spatial_scale()
            r.reset(); prev = None; anchors_before = 0
            for cyc, h in frames[s]:
                if h.shape[0] < 64:
                    r.reset(); prev = None; resets += 1; stream_len.append(cur); cur = 0; continue
                cur += 1
                # the recorder kept every 10th cycle outside short bursts: a missing cycle is just a longer
                # step of the same stream (as in deployment); only a frame without the human ends the stream
                n_before = norm.reanchor_count if norm else 0
                r.begin_profile_window()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                r.stage_world(torch.from_numpy(h).cuda()); out = r.replay_next()
                torch.cuda.synchronize(); wall.append(1e3 * (time.perf_counter() - t0)); n_frames += 1
                prof = r.resolve_profile_window(); ms.append(prof.get("encode_ms", 0.0) + prof.get("decode_ms", 0.0))
                reanchored = norm is not None and norm.reanchor_count > n_before
                if out is not None and prev is not None and cyc == prev + 1 and (s, prev) in gts:
                    disp, matched, n_src = gts[(s, prev)]
                    sel = r.source_selection_indices().cpu().numpy()
                    if n_src == frames_len.get((s, prev), -1):
                        f = r.flow_world()[0].cpu().numpy()
                        e = np.linalg.norm(f - disp[sel], axis=1)[matched[sel]]
                        err.append(e); gn.append(np.linalg.norm(disp[sel], axis=1)[matched[sel]]); after.append(np.full(e.shape, reanchored))
                prev = cyc
            stream_len.append(cur); cur = 0
        e, g, a = np.concatenate(err), np.concatenate(gn), np.concatenate(after); b = np.digitize(g, BINS) - 1
        row = dict(all=float(e.mean() * 1e3), bins=[float(e[b == i].mean() * 1e3) for i in range(len(BINS) - 1)],
                   reanchor_frames_epe=float(e[a].mean() * 1e3) if a.any() else None, other_frames_epe=float(e[~a].mean() * 1e3),
                   reanchors=int(norm.reanchor_count) if norm else 0, reencodes=int(r.reencode_count), frames=n_frames, resets=resets,
                   model_ms_p50=float(np.median(ms)), model_ms_p95=float(np.percentile(ms, 95)), wall_ms_p50=float(np.median(wall)),
                   wall_ms_p95=float(np.percentile(wall, 95)), eval_points=int(e.size), streams=int(sum(1 for x in stream_len if x)), longest_stream=int(max(stream_len)))
        res[name] = row
        print(f"{name:32s} EPE {row['all']:5.2f} mm | bins " + " ".join(f"{x:5.1f}" for x in row["bins"])
              + f" | re-anchors {row['reanchors']:3d} re-encodes {row['reencodes']:3d} / {n_frames} frames | model ms p50 {row['model_ms_p50']:.2f} p95 {row['model_ms_p95']:.2f}"
              + f" | streams {row['streams']} (longest {row['longest_stream']})" + f" | EPE on re-anchor frames {row['reanchor_frames_epe'] if row['reanchor_frames_epe'] is None else round(row['reanchor_frames_epe'], 2)}", flush=True)
        del r; torch.cuda.empty_cache()
    if OUT:
        os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)

frames_len = {}
OUT = None
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="stream the recorded tracked human through the runner, EPE vs pseudo GT")
    ap.add_argument("--recordings", default=REC, help="<scene>/cycle_*.npz of the PerceptiveSafetyFilter recorder")
    ap.add_argument("--gt", default=GT, help="<scene>/gap1/{pair,gt}_*.npz (DifFlow3D-uncertainty cov/human_gt.py)")
    ap.add_argument("--out", default=None, help="optional json output")
    a = ap.parse_args()
    REC, GT, OUT = a.recordings, a.gt, a.out
    # GT pairs index the human points of the recorded cycle in recording order: check the counts once
    for s in SCENES:
        for f in sorted(glob.glob(f"{REC}/{s}/cycle_*.npz")):
            cyc = int(os.path.basename(f)[6:13])
            if os.path.exists(f"{GT}/{s}/gap1/pair_{cyc:07d}.npz"):
                d = np.load(f); frames_len[(s, cyc)] = int((d["track_ids"] == 1).sum())
    main()
