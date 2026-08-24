#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda_utils.h"
#include "voxel_components_gpu.h"

namespace {

__device__ __forceinline__ int find_root(const int *parents, int index) {
    int parent = parents[index];
    while (parent != parents[parent]) {
        parent = parents[parent];
    }
    return parent;
}

__device__ __forceinline__ void unite(int *parents, int first, int second) {
    while (true) {
        const int root_first = find_root(parents, first);
        const int root_second = find_root(parents, second);
        if (root_first == root_second) {
            return;
        }

        const int lower = min(root_first, root_second);
        const int higher = max(root_first, root_second);
        if (atomicCAS(parents + higher, higher, lower) == higher) {
            return;
        }
    }
}

__device__ __forceinline__ int binary_search_key(
    const int64_t *keys,
    int count,
    int64_t query) {
    int lower = 0;
    int upper = count;
    while (lower < upper) {
        const int middle = lower + (upper - lower) / 2;
        if (keys[middle] < query) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower < count && keys[lower] == query ? lower : -1;
}

__device__ __forceinline__ int compare_coordinate(
    const int *coordinates,
    int index,
    int x,
    int y,
    int z) {
    const int value_x = coordinates[index * 3 + 0];
    const int value_y = coordinates[index * 3 + 1];
    const int value_z = coordinates[index * 3 + 2];
    if (value_x != x) {
        return value_x < x ? -1 : 1;
    }
    if (value_y != y) {
        return value_y < y ? -1 : 1;
    }
    if (value_z != z) {
        return value_z < z ? -1 : 1;
    }
    return 0;
}

__device__ __forceinline__ bool contains_coordinate(
    const int *coordinates,
    int count,
    int x,
    int y,
    int z) {
    int lower = 0;
    int upper = count;
    while (lower < upper) {
        const int middle = lower + (upper - lower) / 2;
        if (compare_coordinate(coordinates, middle, x, y, z) < 0) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower < count &&
           compare_coordinate(coordinates, lower, x, y, z) == 0;
}

__global__ void initialize_parents_kernel(int count, int *parents) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        parents[index] = index;
    }
}

__global__ void union_voxel_neighbors_kernel(
    int count,
    const int64_t *__restrict__ sorted_keys,
    const int *__restrict__ shifted_coords,
    const int *__restrict__ extents,
    int *__restrict__ parents) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const int x = shifted_coords[index * 3 + 0];
    const int y = shifted_coords[index * 3 + 1];
    const int z = shifted_coords[index * 3 + 2];
    const int extent_x = extents[0];
    const int extent_y = extents[1];
    const int extent_z = extents[2];
    const int64_t yz_extent = static_cast<int64_t>(extent_y) * extent_z;

    // Query only the lexicographically positive half of the 26-neighborhood.
    // Every undirected edge is therefore visited exactly once.
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (!(dx > 0 || (dx == 0 && dy > 0) ||
                      (dx == 0 && dy == 0 && dz > 0))) {
                    continue;
                }

                const int nx = x + dx;
                const int ny = y + dy;
                const int nz = z + dz;
                if (nx < 0 || ny < 0 || nz < 0 ||
                    nx >= extent_x || ny >= extent_y || nz >= extent_z) {
                    continue;
                }

                const int64_t neighbor_key =
                    static_cast<int64_t>(nx) * yz_extent +
                    static_cast<int64_t>(ny) * extent_z + nz;
                const int neighbor = binary_search_key(
                    sorted_keys, count, neighbor_key);
                if (neighbor >= 0) {
                    unite(parents, index, neighbor);
                }
            }
        }
    }
}

__global__ void compress_and_count_kernel(
    int count,
    int *parents,
    int *component_sizes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const int root = find_root(parents, index);
    parents[index] = root;
    atomicAdd(component_sizes + root, 1);
}

__global__ void component_summary_kernel(
    int count,
    const int *__restrict__ parents,
    const int *__restrict__ component_sizes,
    int *largest_component_size,
    int *__restrict__ statistics) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count && parents[index] == index) {
        atomicMax(largest_component_size, component_sizes[index]);
        atomicMax(statistics + 1, component_sizes[index]);
        atomicAdd(statistics + 0, 1);
    }
}

__global__ void temporal_support_kernel(
    int count,
    const int *__restrict__ parents,
    const int *__restrict__ component_sizes,
    const int *__restrict__ largest_component_size,
    const int *__restrict__ absolute_coords,
    const int *__restrict__ previous_absolute_coords,
    int previous_count,
    int tiny_component_max_voxels,
    int max_small_component_voxels,
    int support_radius_voxels,
    int *__restrict__ supported_counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const int root = parents[index];
    const int size = component_sizes[root];
    const int largest = largest_component_size[0];
    if (size == largest || size <= tiny_component_max_voxels ||
        size > max_small_component_voxels) {
        return;
    }

    const int x = absolute_coords[index * 3 + 0];
    const int y = absolute_coords[index * 3 + 1];
    const int z = absolute_coords[index * 3 + 2];
    for (int dx = -support_radius_voxels;
         dx <= support_radius_voxels; ++dx) {
        for (int dy = -support_radius_voxels;
             dy <= support_radius_voxels; ++dy) {
            for (int dz = -support_radius_voxels;
                 dz <= support_radius_voxels; ++dz) {
                if (contains_coordinate(
                        previous_absolute_coords,
                        previous_count,
                        x + dx,
                        y + dy,
                        z + dz)) {
                    atomicAdd(supported_counts + root, 1);
                    return;
                }
            }
        }
    }
}

__global__ void classify_component_roots_kernel(
    int count,
    const int *__restrict__ parents,
    const int *__restrict__ component_sizes,
    const int *__restrict__ supported_counts,
    const int *__restrict__ largest_component_size,
    int previous_count,
    int tiny_component_max_voxels,
    int max_small_component_voxels,
    float min_supported_fraction,
    bool *__restrict__ keep_mask,
    int *__restrict__ statistics) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || parents[index] != index) {
        return;
    }

    const int size = component_sizes[index];
    const bool is_largest = size == largest_component_size[0];
    bool keep = true;
    if (!is_largest && size <= tiny_component_max_voxels) {
        keep = false;
        atomicAdd(statistics + 2, 1);
    } else if (!is_largest && size <= max_small_component_voxels) {
        atomicAdd(statistics + 3, 1);
        atomicAdd(statistics + 9, size);
        if (previous_count == 0) {
            atomicAdd(statistics + 10, 1);
        } else {
            const int supported = supported_counts[index];
            atomicAdd(statistics + 8, supported);
            keep = static_cast<float>(supported) >=
                   min_supported_fraction * static_cast<float>(size);
            atomicAdd(statistics + (keep ? 4 : 5), 1);
        }
    }

    keep_mask[index] = keep;
    if (!keep) {
        atomicAdd(statistics + 6, 1);
        atomicAdd(statistics + 7, size);
    }
}

__global__ void expand_component_keep_mask_kernel(
    int count,
    const int *__restrict__ parents,
    bool *__restrict__ keep_mask) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        keep_mask[index] = keep_mask[parents[index]];
    }
}

}  // namespace

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
    cudaStream_t stream) {
    const int blocks = DIVUP(count, THREADS_PER_BLOCK);
    cudaMemsetAsync(component_sizes, 0, count * sizeof(int), stream);
    cudaMemsetAsync(supported_counts, 0, count * sizeof(int), stream);
    cudaMemsetAsync(largest_component_size, 0, sizeof(int), stream);
    cudaMemsetAsync(statistics, 0, 11 * sizeof(int), stream);

    initialize_parents_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        count, parents);
    union_voxel_neighbors_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        count, sorted_keys, shifted_coords, extents, parents);
    compress_and_count_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        count, parents, component_sizes);
    component_summary_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        count, parents, component_sizes, largest_component_size, statistics);
    if (previous_count > 0 &&
        max_small_component_voxels > tiny_component_max_voxels) {
        temporal_support_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            count,
            parents,
            component_sizes,
            largest_component_size,
            absolute_coords,
            previous_absolute_coords,
            previous_count,
            tiny_component_max_voxels,
            max_small_component_voxels,
            support_radius_voxels,
            supported_counts);
    }
    classify_component_roots_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        count,
        parents,
        component_sizes,
        supported_counts,
        largest_component_size,
        previous_count,
        tiny_component_max_voxels,
        max_small_component_voxels,
        min_supported_fraction,
        keep_mask,
        statistics);
    expand_component_keep_mask_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        count, parents, keep_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
