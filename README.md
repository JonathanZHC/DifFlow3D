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
│   ├── docker_build.sh         # build difflow3d:latest
│   ├── run_simulation.sh       # create/reuse dev container + run benchmark
│   ├── run_rviz.sh             # launch RViz in the same container
│   ├── build_pointnet2_ops.sh  # rebuild native CUDA extension
│   ├── test_voxel_difflow.py
│   ├── test_anchor_motion_ops.py
│   ├── test_superquadrics.py
│   ├── test_runtime_ops.py
│   ├── benchmark_optimizations.py
│   └── benchmark_recovery.py
├── configs/config.yaml
├── rviz/voxel_difflow.rviz
├── checkpoints/
└── Dockerfile
```


## Docker quick start

The host-side helper scripts keep the development checkout mounted at `/workspace`, so source/config changes are visible without rebuilding the Docker image. The native PointNet2 extension is rebuilt by `run_simulation.sh` only when it is missing or older than its C++/CUDA sources.

Build the image once:

```bash
bash scripts/docker_build.sh
```

Run the configured benchmark/simulation:

```bash
bash scripts/run_simulation.sh
```

For live visualization, use two host terminals. The first command enables the benchmark publisher, runs at sensor rate, and keeps the ROS publisher alive until `Ctrl+C`:

```bash
# terminal 1
bash scripts/run_simulation.sh --rviz --realtime --rviz-hold-seconds -1

# terminal 2
bash scripts/run_rviz.sh
```

For timing-only runs, keep RViz and real-time pacing disabled:

```bash
bash scripts/run_simulation.sh --no-rviz --no-realtime
```

The scripts reuse a container named `difflow3d`. Remove it when you want a fresh container:

```bash
docker rm -f difflow3d
```

Optional environment overrides are `DIFFLOW_IMAGE` (default `difflow3d:latest`), `DIFFLOW_CONTAINER` (default `difflow3d`), `DIFFLOW_CONFIG` (default `configs/config.yaml`), and `DIFFLOW_RVIZ_CONFIG` (default `rviz/voxel_difflow.rviz`).

If RViz cannot connect to the display, verify that `DISPLAY` is set on the host and that X11/Xwayland allows the current local user.

## Production pipeline

```text
raw world cloud
  -> first metric voxel downsample
  -> adaptive voxel-2
  -> optional voxel connected-component outlier filter
  -> exact-count selection -> 2048 anchors
  -> frozen spatial canonicalization
  -> encode each frame once
  -> CUDA-Graph pair decode
  -> world-space sparse flow
  -> optional velocity-only KF + track-aware state transport
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

The KF state-transport neighborhood reuses `recovery.softmax_sigma_m`, `recovery.local_radius_sigma`, and `recovery.local_hash_size_factor`; there is no separate transport-tuning block.

The Euclidean KNN path is fixed to the PyTorch GEMM + `topk` implementation because it measured faster than the experimental custom KNN kernel on the target RTX 5090. Neural inference is FP32 with TF32 enabled; the slower BF16 experiment is not part of the deployment API.

## Fixed-count preprocessing

Let `K = preprocessing.fps_points` and `C = ceil(second_candidate_ratio * K)`:

```text
N1 < K       -> deterministic repeat to K
N1 = K       -> direct
K < N1 <= C  -> exact-count selection to K
N1 > C       -> adaptive voxel-2
                 optional component filter
                 N2 < K  -> repeat
                 N2 = K  -> direct
                 N2 > K  -> exact-count selection
```

`final_selection: uniform` selects deterministic evenly spaced representatives from the voxel-key-ordered candidate cloud. It preserves exactly `K` points and avoids the iterative runtime cost of FPS. `fps` remains available as a validation/reference option.

Voxel-2 resolution is calibrated once and then frozen. Spatial scale is also calibrated once and frozen until the preprocessing calibration is explicitly reset.

The optional outlier filter operates after voxel-2 and before exact-count
selection/FPS. It uses the following rule. Let `N` be the current number of
unique voxel-2 representatives, `H = tiny_component_max_voxels`, and

\[
S = \max\left(H,\left\lceil
\texttt{max\_small\_component\_fraction}\,N
\right\rceil\right).
\]

The current voxels are split into sparse 26-neighbor connected components. All
components tied for largest are retained. Every other component of size `s` is
classified as follows:

1. `s <= H`: remove immediately.
2. `H < s <= S`: retain only when its previous-frame supported-voxel fraction
   is at least `min_supported_fraction`.
3. `s > S`: retain.

A current voxel is temporally supported when the unfiltered previous-frame
voxel set contains at least one voxel within Chebyshev distance
`support_radius_voxels`. Thus a value of `1` checks the surrounding 3 x 3 x 3
voxel neighborhood and tolerates one-voxel motion or quantization jitter. On the
first frame, temporal evidence is unavailable: medium components are retained,
while the tiny-component rule still applies. History always contains only the
immediately preceding unfiltered voxel-2 observations and is cleared when a
frame bypasses voxel-2 or the streaming runner is reset.

The preprocessing API does not receive track IDs, so the implementation treats
the complete cloud as one virtual object. Temporal matching uses absolute
world-space voxel coordinates; upstream world-frame alignment is therefore
preserved naturally. Multiple large occlusion-separated blocks remain valid
because every largest component and every component above `S` is retained.

When enabled, the benchmarks print the component count, largest block size,
tiny blocks removed, temporal candidates, supported/rejected candidates,
removed blocks/voxels, temporal support fraction, and filter time. CUDA writes
these statistics to a fixed-size buffer; it is read only after an already
required benchmark synchronization, so reporting does not add a synchronization
to the measured filter path.

The existing repeat path handles a retained count below `K`; the filter has no
separate point-count fallback.

The filter is part of the bundled PointNet2 extension and adds no Python package
dependency. Rebuild the extension after pulling these sources:

```bash
bash scripts/build_pointnet2_ops.sh
python3 scripts/test_voxel_outlier_filter.py
```

## Spatial canonicalization

Set `preprocessing.auto_spatial_scale.enable` to `true` to fit the scale from
`target_model_volume`. Set it to `false` to use `fixed_spatial_scale` directly.

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

## Velocity-only temporal Kalman filter

The optional temporal filter stays entirely in the velocity domain. Its state is `[vx, vy, vz]` with diagonal covariance and the random-walk model

\[
v_k = v_{k-1} + w_k, \qquad w_k \sim \mathcal N(0,Q_v).
\]

There is no acceleration state, acceleration measurement, or velocity-to-acceleration finite difference. The current DifFlow displacement is converted once to the velocity measurement `z_k = flow_k / dt`. With `kalman.enabled=false`, the temporal stage is bypassed and the raw velocity is used unchanged.

```yaml
motion_estimation:
  kalman:
    enabled: true
    process_velocity_std_mps: 0.15
    measurement_noise_std_mps: 0.10
    initial_velocity_std_mps: 0.30
    min_innovation_variance: 1.0e-6

# KF state transport reuses recovery.softmax_sigma_m,
# recovery.local_radius_sigma, and recovery.local_hash_size_factor.

```

The track-aware CUDA transport moves the previous filtered state to the current anchor set. It transports first/second Gaussian moments so local velocity variation is reflected in the transported covariance. Every finite DifFlow velocity measurement is fused by the standard Kalman update. No legacy constant-acceleration mode or second-order motion parameters are part of the deployment configuration.

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

The script removes stale local build products, builds the extension in-place,
verifies that the loaded `.so` comes from the current checkout, and checks the
required recovery, motion, and voxel-component symbols.

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

From the host, the recommended entry point is:

```bash
bash scripts/run_simulation.sh
```

Inside an already-running container, the direct command is still available:

```bash
python3 scripts/test_voxel_difflow.py --config configs/config.yaml
```

With detailed profiling enabled, the benchmark reports first voxel, voxel-2,
voxel outlier filtering, final selection, scale/staging, encode, decode,
velocity-KF filtering (when enabled), dense recovery, and overall latency.

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
