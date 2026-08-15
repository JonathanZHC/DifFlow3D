# DifFlow3D deployment cheatsheet

## Build

```bash
cd /workspace
bash scripts/build_pointnet2_ops.sh
```

## Main benchmark

```bash
python3 scripts/test_voxel_difflow.py --config configs/config.yaml
```

## Recovery kernel validation

```bash
python3 scripts/test_runtime_ops.py \
  --device cuda:0 \
  --points 2048 \
  --queries 96000 \
  --sigma 0.025 \
  --radius-sigma 4
```

## Recovery sweep

```bash
python3 scripts/benchmark_recovery.py --config configs/config.yaml
```

## Iteration sweep

```bash
python3 scripts/benchmark_optimizations.py --config configs/config.yaml
```

## Current deployment settings

```yaml
runtime:
  enable_tf32: true

model:
  iterations: {coarse: 4, middle: 2, fine: 2}

preprocessing:
  fps_points: 2048
  second_candidate_ratio: 1.1
  final_selection: uniform
  auto_spatial_scale: true
  target_model_volume: 2.0

recovery:
  softmax_sigma_m: 0.025
  backend: local
  local_radius_sigma: 4.0
  local_hash_size_factor: 4.0

rviz:
  enabled: false
```

## Recovery backends

```text
global  exact CUDA reference
local   radius-local CUDA hash grid; exact-global fallback if empty
torch   exact chunked PyTorch fallback
auto    global if available, else torch
```

`softmax_sigma_m` is always in world metres.
