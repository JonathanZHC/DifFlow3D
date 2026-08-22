#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

void gaussian_softmax_transport_local_track_aware_kernel_launcher(
    int query_count,
    int anchor_count,
    int channels,
    int hash_size,
    float sigma,
    float radius,
    float cell_size,
    bool global_same_track_fallback,
    const float *queries,
    const float *anchors,
    const float *anchor_values,
    const int *query_track_ids,
    const int *anchor_track_ids,
    const int *hash_heads,
    const int *anchor_next,
    const int *anchor_cells,
    float *output_values,
    int *support_counts,
    cudaStream_t stream);

void anchor_kalman_update_kernel_launcher(
    int count,
    float dt,
    float process_velocity_variance,
    float measurement_variance,
    float initial_velocity_variance,
    float min_innovation_variance,
    const float *current_flow,
    const float *transported_previous_state6,
    const int *support_counts,
    float *output_state6,
    cudaStream_t stream);
