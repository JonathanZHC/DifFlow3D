#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from difflow3d.config import load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from difflow3d.testing.superquadrics_benchmark import run
    run(load_config(args.config))
