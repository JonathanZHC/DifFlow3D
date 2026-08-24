# PointNet++ / recovery CUDA extension

This directory contains the native operators required by DifFlow3D inference:

- PointNet++ sampling/grouping/interpolation kernels used by the network.
- exact global Gaussian-softmax dense recovery.
- radius-local hash-grid Gaussian-softmax dense recovery.
- sparse 26-neighbor voxel components with one-frame temporal support.

The model's Euclidean KNN path uses PyTorch GEMM + `topk`; the slower experimental custom KNN extension is intentionally not built.

Build from the repository root:

```bash
bash scripts/build_pointnet2_ops.sh
```

The extension is produced as `pointnet2_cuda*.so` next to `pointnet2_utils.py`. Build products are git-ignored and should not be committed.
