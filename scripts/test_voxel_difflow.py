#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from difflow3d.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "config.yaml",
    )
    parser.add_argument(
        "--rviz",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override rviz.enabled from the YAML config.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override benchmark.realtime from the YAML config.",
    )
    parser.add_argument(
        "--rviz-hold-seconds",
        type=float,
        default=None,
        help="Override rviz.hold_seconds; use -1 to hold until Ctrl+C.",
    )
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.rviz is not None:
        config["rviz"]["enabled"] = bool(args.rviz)
    if args.realtime is not None:
        config["benchmark"]["realtime"] = bool(args.realtime)
    if args.rviz_hold_seconds is not None:
        config["rviz"]["hold_seconds"] = float(args.rviz_hold_seconds)
    return config


if __name__ == "__main__":
    args = parse_args()
    from difflow3d.testing.voxel_benchmark import run

    run(apply_overrides(load_config(args.config), args))
