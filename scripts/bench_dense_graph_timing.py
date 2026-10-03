# Exploratory benchmark (2026-10-02): runs inside the perceptive-safety-filter container with /tmp/real = ~/datasets/scene_flow/real (human pairs + pseudo GT). Monkeypatches the model; not wired into the pipeline.
import sys, json, gc, os
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, "/opt/DifFlow3D"); sys.path.insert(0, "/tmp")
import torch
os.environ.setdefault("TOP", "2048")
import fast8192 as F                      # patches (import only; CONFIGS loop is guarded below)
from difflow3d.config import load_config, parse_iteration_schedule
from difflow3d.testing.voxel_benchmark import run
base = load_config(Path("/opt/DifFlow3D/configs/config.yaml"))
out = Path("/tmp/graph_bench"); out.mkdir(exist_ok=True)
runs = [("8192_A", 8192, (4, 2, 2), ()), ("8192_B", 8192, (4, 2, 2), ("rand",)), ("8192_B_cos", 8192, (4, 2, 2), ("rand", "cos")),
        ("8192_B_fine1", 8192, (4, 2, 1), ("rand",)), ("4096_A", 4096, (4, 2, 2), ()), ("4096_B", 4096, (4, 2, 2), ("rand",)), ("4096_B_fine1", 4096, (4, 2, 1), ("rand",))]
for name, N, it, flags in runs:
    F.apply(flags); F._cache.clear()
    cfg = deepcopy(base)
    cfg["preprocessing"]["fps_points"] = N
    cfg["model"]["iterations"] = parse_iteration_schedule({"coarse": it[0], "middle": it[1], "fine": it[2]})
    cfg["preprocessing"]["final_selection"] = "uniform"
    cfg["benchmark"]["frames"] = 40; cfg["rviz"]["enabled"] = False
    cfg["benchmark"]["json_output"] = str(out / f"{name}.json")
    try: run(cfg)
    except Exception as e: print("FAILED", name, str(e)[:300])
    gc.collect(); torch.cuda.empty_cache()
print("\n=== GRAPH SUMMARY (median ms) ===")
for name, *_ in runs:
    p = out / f"{name}.json"
    if not p.exists(): continue
    d = json.loads(p.read_text()); t = d["timing_ms"]; a = d["accuracy"]["first_flow_epe_m"]
    print(f"{name:14s}: encode {t['runner_encode_ms']['median']:.2f} decode {t['runner_decode_ms']['median']:.2f} model {t['runner_model_total_ms']['median']:.2f} overall {t['overall_wall_ms']['median']:.2f} | synthetic EPE {a['mean']*1e3:.2f} mm")
