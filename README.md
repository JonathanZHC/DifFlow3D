# DifFlow3D — inference-only deployment repository

This checkout contains only the code needed for online DifFlow3D inference, runtime preprocessing, CUDA-Graph execution, dense flow recovery, synthetic validation, and RViz visualization. Training/data-preparation code and generated build/profiling artifacts are intentionally excluded.

The network remains checkpoint-compatible with:

```text
checkpoints/model_difflow_355_0.0114.pth
```

## Layout

```text
DifFlow3D/
├── difflow3d/
│   ├── model/                 # checkpoint-compatible DifFlow3D network
│   ├── runtime/               # streaming inference, preprocessing, recovery
│   ├── ops/pointnet2/         # PointNet++ and recovery CUDA extension
│   └── testing/               # synthetic benchmark/metrics/RViz helpers
├── scripts/
│   ├── test_voxel_difflow.py
│   ├── test_superquadrics.py
│   ├── test_runtime_ops.py
│   ├── benchmark_optimizations.py
│   ├── benchmark_recovery.py
│   └── build_pointnet2_ops.sh
├── configs/config.yaml
├── rviz/voxel_difflow.rviz
├── checkpoints/
└── Dockerfile
```

## Production pipeline

```text
raw world cloud
  -> first metric voxel downsample
  -> adaptive voxel-2
  -> exact-count selection -> 2048 anchors
  -> frozen spatial canonicalization
  -> encode each frame once
  -> CUDA-Graph pair decode
  -> world-space sparse flow
  -> dense Gaussian recovery
```

The measured deployment choices are intentionally simple:

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

The Euclidean KNN path is fixed to the PyTorch GEMM + `topk` implementation because it measured faster than the experimental custom KNN kernel on the target RTX 5090. Neural inference is FP32 with TF32 enabled; the slower BF16 experiment is not part of the deployment API.

## Fixed-count preprocessing

Let `K = preprocessing.fps_points` and `C = ceil(second_candidate_ratio * K)`:

```text
N1 < K       -> deterministic repeat to K
N1 = K       -> direct
K < N1 <= C  -> exact-count selection to K
N1 > C       -> adaptive voxel-2
                 N2 < K  -> repeat
                 N2 = K  -> direct
                 N2 > K  -> exact-count selection
```

`final_selection: uniform` selects deterministic evenly spaced representatives from the voxel-key-ordered candidate cloud. It preserves exactly `K` points and avoids the iterative runtime cost of FPS. `fps` remains available as a validation/reference option.

Voxel-2 resolution is calibrated once and then frozen. Spatial scale is also calibrated once and frozen until the preprocessing calibration is explicitly reset.

## Spatial canonicalization

With automatic scaling enabled:

\[
s = \left(\frac{V_{target}}{V_{world}}\right)^{1/3}.
\]

Model inputs use:

\[
P_m=sP_w,
\]

and predicted displacement returns to world units with:

\[
\Delta P_w=\Delta P_m/s.
\]

Dense recovery always operates in world coordinates. Therefore `recovery.softmax_sigma_m` is a real metric bandwidth in metres and is never scaled by the model canonicalization.

## Recurrent iteration schedule

The inference model accepts independent refinement counts:

```yaml
model:
  iterations:
    coarse: 4   # 512 points
    middle: 2   # 1024 points
    fine: 2     # 2048 points
```

A scalar is still accepted and maps to the same count at all three levels. Iteration count changes execution only; checkpoint parameters are unchanged.

## Dense recovery

The exact global reference is:

\[
w_i(q)=\frac{\exp(-\|q-a_i\|^2/(2\sigma^2))}
{\sum_j \exp(-\|q-a_j\|^2/(2\sigma^2))},
\qquad
f(q)=\sum_i w_i(q)f_i.
\]

Supported backends:

```yaml
recovery:
  backend: local  # auto | global | local | torch
```

- `global`: exact all-anchor CUDA warp-per-query softmax.
- `local`: radius-truncated CUDA hash-grid softmax. The radius is `local_radius_sigma * softmax_sigma_m`; a query with no in-radius anchor falls back to the exact global kernel.
- `torch`: chunked exact PyTorch reference/fallback.
- `auto`: exact CUDA `global` when available, otherwise `torch`. It never silently selects the approximate local path.

The current 4σ local path is the measured-fast deployment default. The exact global backend is retained as the numerical baseline.

## Build the CUDA extension

The source tree already uses modern PyTorch/CUDA APIs; no THC compatibility patching is required.

```bash
cd /workspace
bash scripts/build_pointnet2_ops.sh
```

The script removes stale local build products, builds the extension in-place, verifies that the loaded `.so` comes from the current checkout, and checks the required global/local recovery symbols.

## Validate runtime CUDA ops

```bash
python3 scripts/test_runtime_ops.py \
  --device cuda:0 \
  --points 2048 \
  --queries 96000 \
  --sigma 0.025 \
  --radius-sigma 4
```

This compares exact CUDA recovery against the PyTorch reference and reports local approximation error, neighbor counts, fallback ratio, and runtime.

## Main benchmark

```bash
python3 scripts/test_voxel_difflow.py \
  --config configs/config.yaml
```

With detailed profiling enabled, the benchmark reports first voxel, voxel-2, final selection, scale/staging, encode, decode, dense recovery, and overall latency.

For absolute deployment timing, use:

```yaml
profiling:
  detailed_runtime_breakdown: false

rviz:
  enabled: false
```

## Optional sweeps

Iteration schedule:

```bash
python3 scripts/benchmark_optimizations.py --config configs/config.yaml
```

Dense recovery:

```bash
python3 scripts/benchmark_recovery.py --config configs/config.yaml
```

The recovery sweep compares exact global, local 5σ, and local 4σ while holding the model/preprocessing configuration fixed.

## Repository hygiene

Generated native objects, `.so` files, Python caches, and benchmark outputs are ignored by git and are not part of the source archive. Rebuild the CUDA extension after a fresh checkout.
