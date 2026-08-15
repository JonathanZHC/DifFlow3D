#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

// Exact global warp-per-query Gaussian softmax.
void gaussian_softmax_recovery_kernel_launcher(
    int query_count,
    int anchor_count,
    float sigma,
    const float *queries,
    const float *anchors,
    const float *anchor_flow,
    float *output_flow,
    cudaStream_t stream);

// Build the fixed-size anchor hash grid used by local recovery.
void gaussian_recovery_hash_build_kernel_launcher(
    int anchor_count,
    int hash_size,
    float cell_size,
    const float *anchors,
    int *hash_heads,
    int *anchor_next,
    int *anchor_cells,
    cudaStream_t stream);

// Radius-local Gaussian softmax. Queries with no in-radius anchor fall back to
// the exact global softmax. local_neighbor_counts==0 therefore identifies a
// fallback query.
void gaussian_softmax_recovery_local_kernel_launcher(
    int query_count,
    int anchor_count,
    int hash_size,
    float sigma,
    float radius,
    float cell_size,
    const float *queries,
    const float *anchors,
    const float *anchor_flow,
    const int *hash_heads,
    const int *anchor_next,
    const int *anchor_cells,
    float *output_flow,
    int *local_neighbor_counts,
    cudaStream_t stream);
