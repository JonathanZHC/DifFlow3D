# DifFlow3D deployment cheatsheet

## Docker build

```bash
bash scripts/docker_build.sh
```

## Run benchmark / simulation

```bash
bash scripts/run_simulation.sh
```

## Run simulation with RViz publishing

Terminal 1:

```bash
bash scripts/run_simulation.sh --rviz --realtime --rviz-hold-seconds -1
```

Terminal 2:

```bash
bash scripts/run_rviz.sh
```

## Rebuild native CUDA extension manually

```bash
bash scripts/build_pointnet2_ops.sh
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

## Velocity-KF / temporal CUDA validation

```bash
python3 scripts/test_anchor_motion_ops.py
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
  outlier_filter:
    enabled: true
    tiny_component_max_voxels: 2
    max_small_component_fraction: 0.005
    support_radius_voxels: 1
    min_supported_fraction: 0.30
  auto_spatial_scale:
    enable: true
    target_model_volume: 2.0
    fixed_spatial_scale: 1.0

recovery:
  softmax_sigma_m: 0.025
  backend: local
  local_radius_sigma: 4.0
  local_hash_size_factor: 4.0

motion_estimation:
  kalman:
    enabled: true
    process_velocity_std_mps: 0.15
    measurement_noise_std_mps: 0.10
    initial_velocity_std_mps: 0.30

rviz:
  enabled: false
```

KF state transport reuses the spatial neighborhood parameters under `recovery`; no separate transport parameters are configured.

## Recovery backends

```text
global  exact CUDA reference
local   radius-local CUDA hash grid; exact-global fallback if empty
torch   exact chunked PyTorch fallback
auto    global if available, else torch
```

`softmax_sigma_m` is always in world metres.
