#ifndef _VOXEL_COMPONENTS_GPU_H
#define _VOXEL_COMPONENTS_GPU_H

#include <cuda_runtime.h>
#include <cstdint>

void voxel_component_filter_kernel_launcher(
    int count,
    const int64_t *sorted_keys,
    const int *shifted_coords,
    const int *absolute_coords,
    const int *extents,
    const int *previous_absolute_coords,
    int previous_count,
    int tiny_component_max_voxels,
    int max_small_component_voxels,
    int support_radius_voxels,
    float min_supported_fraction,
    int *parents,
    int *component_sizes,
    int *supported_counts,
    int *largest_component_size,
    bool *keep_mask,
    int *statistics,
    cudaStream_t stream);

#endif
