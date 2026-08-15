#!/usr/bin/env python3
"""Run the 2048-point accuracy/runtime profiles declared in config.yaml."""
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

from difflow3d.config import load_config, parse_iteration_schedule
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
        help="Optional comma-separated subset of optimization_sweep profile names.",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    sweep = base.get("optimization_sweep", {})
    profiles = list(sweep.get("profiles", []))
    if not profiles:
        raise ValueError("config.yaml contains no optimization_sweep.profiles")
    selected = {name for name in args.profiles.split(",") if name}
    if selected:
        profiles = [p for p in profiles if p.get("name") in selected]
    if not profiles:
        raise ValueError("No optimization profiles selected.")

    fixed_points = int(base["preprocessing"]["fps_points"])
    output_dir = Path(base["_repo_root"]) / "outputs" / "optimization_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []

    for profile in profiles:
        name = str(profile["name"])
        print("\n" + "#" * 104)
        print(f"Optimization profile: {name} (fixed model points={fixed_points})")
        print("#" * 104)
        cfg = deepcopy(base)

        if "fps_points" in profile and int(profile["fps_points"]) != fixed_points:
            raise ValueError(
                f"Profile {name!r} changes fps_points. This sweep intentionally "
                f"fixes the point count at {fixed_points}."
            )
        cfg["preprocessing"]["fps_points"] = fixed_points
        cfg["model"]["iterations"] = parse_iteration_schedule(profile["iterations"])
        cfg["preprocessing"]["final_selection"] = str(
            profile.get("final_selection", cfg["preprocessing"].get("final_selection", "fps"))
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
    print("Optimization sweep summary")
    print("=" * 104)
    for path in result_paths:
        data = json.loads(path.read_text())
        timing = data["timing_ms"]
        accuracy = data["accuracy"]
        overall = timing["overall_wall_ms"]["median"]
        runner = timing["runner_model_total_ms"]["median"]
        selection = timing["runner_final_selection_ms"]["median"]
        recovery = timing["recovery_ms"]["median"]
        epe = accuracy["first_flow_epe_m"]["mean"]
        iters = data["iterations"]
        print(
            f"{path.stem:28s} "
            f"iters={iters['coarse']}/{iters['middle']}/{iters['fine']}  "
            f"select={data['final_selection']:7s}  "
            f"overall={overall:7.3f} ms  runner={runner:7.3f}  "
            f"select_ms={selection:6.3f}  recovery={recovery:7.3f}  "
            f"first-EPE={epe:.6f} m"
        )


if __name__ == "__main__":
    main()
