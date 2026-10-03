# Exploratory (2026-10-02): fork-model monkeypatches evaluated inside the perceptive-safety-filter container (/tmp/real = ~/datasets/scene_flow/real). Not wired into the pipeline.
import sys, json, gc
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, "/opt/DifFlow3D"); sys.path.insert(0, "/tmp")
import torch
import algo8192 as A                       # patches applied at import
import fast8192 as F
from difflow3d.config import load_config, parse_iteration_schedule
from difflow3d.testing.voxel_benchmark import run
from difflow3d.runtime import inference as INF
base = load_config(Path("/opt/DifFlow3D/configs/config.yaml"))
out = Path("/tmp/graph_bench2"); out.mkdir(exist_ok=True)
orig_init = INF.DifFlow3DInference.__init__
def init(self, config):                    # the two-resolution patch needs the model instance
    orig_init(self, config); A.MODEL[0] = self.model
INF.DifFlow3DInference.__init__ = init
import os
runs = [("6144_Base", 6144, A.VARIANTS[0][1]), ("6144_H", 6144, A.VARIANTS[1][1])] if os.environ.get("ONLY6144") else [("8192_Base", 8192, A.VARIANTS[0][1]), ("8192_H", 8192, A.VARIANTS[1][1]), ("8192_M", 8192, A.VARIANTS[2][1]), ("8192_MH", 8192, A.VARIANTS[3][1]), ("8192_MS", 8192, A.VARIANTS[4][1]),
        ("4096_Base", 4096, A.VARIANTS[0][1]), ("4096_H", 4096, A.VARIANTS[1][1]), ("4096_MS", 4096, A.VARIANTS[4][1])]
for name, N, cfg in runs:
    A.CFG.update(cfg); F._cache.clear(); A.G.clear()
    c = deepcopy(base)
    c["preprocessing"]["fps_points"] = N; c["model"]["iterations"] = parse_iteration_schedule({"coarse": 4, "middle": 2, "fine": 2})
    c["preprocessing"]["final_selection"] = "uniform"; c["benchmark"]["frames"] = 40; c["rviz"]["enabled"] = False
    c["benchmark"]["json_output"] = str(out / f"{name}.json")
    try: run(c)
    except Exception as e: print("FAILED", name, repr(e)[:300])
    gc.collect(); torch.cuda.empty_cache()
print("\n=== GRAPH SUMMARY (median ms) ===")
for name, *_ in runs:
    p = out / f"{name}.json"
    if not p.exists(): continue
    d = json.loads(p.read_text()); t = d["timing_ms"]; a = d["accuracy"]["first_flow_epe_m"]
    print(f"{name:10s}: encode {t['runner_encode_ms']['median']:.2f} decode {t['runner_decode_ms']['median']:.2f} model {t['runner_model_total_ms']['median']:.2f} overall {t['overall_wall_ms']['median']:.2f} | synthetic EPE {a['mean']*1e3:.2f} mm")
