#!/usr/bin/env python3
"""Compare dense-recovery backends with the same fixed DifFlow configuration."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from difflow3d.config import load_config
from difflow3d.testing.voxel_benchmark import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "config.yaml",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Optional comma-separated subset of recovery_sweep profile names.",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    sweep = base.get("recovery_sweep", {})
    profiles = list(sweep.get("profiles", []))
    if not profiles:
        raise ValueError("config.yaml contains no recovery_sweep.profiles")
    selected = {name for name in args.profiles.split(",") if name}
    if selected:
        profiles = [p for p in profiles if p.get("name") in selected]
    if not profiles:
        raise ValueError("No recovery profiles selected.")

    output_dir = Path(base["_repo_root"]) / "outputs" / "recovery_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []

    for profile in profiles:
        name = str(profile["name"])
        print("\n" + "#" * 104)
        print(f"Recovery profile: {name}")
        print("#" * 104)
        cfg = deepcopy(base)
        cfg["recovery"]["backend"] = str(profile["backend"])
        cfg["recovery"]["report_local_stats"] = str(profile["backend"]) == "local"
        if "local_radius_sigma" in profile:
            cfg["recovery"]["local_radius_sigma"] = float(
                profile["local_radius_sigma"]
            )
        if "local_hash_size_factor" in profile:
            cfg["recovery"]["local_hash_size_factor"] = float(
                profile["local_hash_size_factor"]
            )
        cfg["benchmark"]["frames"] = int(
            sweep.get("frames", cfg["benchmark"]["frames"])
        )
        cfg["rviz"]["enabled"] = False
        output = output_dir / f"{name}.json"
        cfg["benchmark"]["json_output"] = str(output)
        run(cfg)
        result_paths.append(output)
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 104)
    print("Recovery sweep summary")
    print("=" * 104)
    for path in result_paths:
        data = json.loads(path.read_text())
        timing = data["timing_ms"]
        accuracy = data["accuracy"]
        local_stats = data.get("local_recovery_frame_stats", [])
        if local_stats:
            neigh = sum(x["mean"] for x in local_stats) / len(local_stats)
            empty = sum(x["empty_ratio"] for x in local_stats) / len(local_stats)
            local_text = f" neighbors={neigh:6.1f} empty={empty:.5f}"
        else:
            local_text = ""
        print(
            f"{path.stem:24s} "
            f"recovery={timing['recovery_ms']['median']:7.3f} ms  "
            f"overall={timing['overall_wall_ms']['median']:7.3f} ms  "
            f"first-EPE={accuracy['first_flow_epe_m']['mean']:.6f} m"
            f"{local_text}"
        )


if __name__ == "__main__":
    main()
