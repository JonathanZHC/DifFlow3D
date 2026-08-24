#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <climits>

#include "voxel_components_gpu.h"

namespace {

void check_cuda_contiguous(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void voxel_component_filter_wrapper(
    at::Tensor sorted_keys,
    at::Tensor shifted_coords,
    at::Tensor absolute_coords,
    at::Tensor extents,
    at::Tensor previous_absolute_coords,
    int64_t tiny_component_max_voxels,
    int64_t max_small_component_voxels,
    int64_t support_radius_voxels,
    double min_supported_fraction,
    at::Tensor parents,
    at::Tensor component_sizes,
    at::Tensor supported_counts,
    at::Tensor largest_component_size,
    at::Tensor keep_mask,
    at::Tensor statistics) {
    check_cuda_contiguous(sorted_keys, "sorted_keys");
    check_cuda_contiguous(shifted_coords, "shifted_coords");
    check_cuda_contiguous(absolute_coords, "absolute_coords");
    check_cuda_contiguous(extents, "extents");
    check_cuda_contiguous(previous_absolute_coords, "previous_absolute_coords");
    check_cuda_contiguous(parents, "parents");
    check_cuda_contiguous(component_sizes, "component_sizes");
    check_cuda_contiguous(supported_counts, "supported_counts");
    check_cuda_contiguous(largest_component_size, "largest_component_size");
    check_cuda_contiguous(keep_mask, "keep_mask");
    check_cuda_contiguous(statistics, "statistics");

    TORCH_CHECK(sorted_keys.scalar_type() == at::kLong, "sorted_keys must be int64");
    TORCH_CHECK(shifted_coords.scalar_type() == at::kInt, "shifted_coords must be int32");
    TORCH_CHECK(absolute_coords.scalar_type() == at::kInt, "absolute_coords must be int32");
    TORCH_CHECK(extents.scalar_type() == at::kInt, "extents must be int32");
    TORCH_CHECK(previous_absolute_coords.scalar_type() == at::kInt, "previous_absolute_coords must be int32");
    TORCH_CHECK(parents.scalar_type() == at::kInt, "parents must be int32");
    TORCH_CHECK(component_sizes.scalar_type() == at::kInt, "component_sizes must be int32");
    TORCH_CHECK(supported_counts.scalar_type() == at::kInt, "supported_counts must be int32");
    TORCH_CHECK(largest_component_size.scalar_type() == at::kInt, "largest_component_size must be int32");
    TORCH_CHECK(keep_mask.scalar_type() == at::kBool, "keep_mask must be bool");
    TORCH_CHECK(statistics.scalar_type() == at::kInt, "statistics must be int32");

    TORCH_CHECK(sorted_keys.dim() == 1, "sorted_keys must have shape [N]");
    TORCH_CHECK(
        shifted_coords.dim() == 2 && shifted_coords.size(1) == 3,
        "shifted_coords must have shape [N,3]");
    TORCH_CHECK(
        absolute_coords.dim() == 2 && absolute_coords.size(1) == 3,
        "absolute_coords must have shape [N,3]");
    TORCH_CHECK(
        previous_absolute_coords.dim() == 2 && previous_absolute_coords.size(1) == 3,
        "previous_absolute_coords must have shape [M,3]");
    TORCH_CHECK(extents.numel() == 3, "extents must contain three values");

    const int64_t count64 = sorted_keys.size(0);
    TORCH_CHECK(count64 > 0, "voxel component filtering requires at least one voxel");
    TORCH_CHECK(count64 <= INT_MAX, "too many voxels for int32 component workspaces");
    TORCH_CHECK(shifted_coords.size(0) == count64, "key/coordinate count mismatch");
    TORCH_CHECK(absolute_coords.size(0) == count64, "absolute coordinate count mismatch");
    TORCH_CHECK(previous_absolute_coords.size(0) <= INT_MAX, "too many history voxels");
    TORCH_CHECK(parents.numel() >= count64, "parents workspace is too small");
    TORCH_CHECK(component_sizes.numel() >= count64, "component_sizes workspace is too small");
    TORCH_CHECK(supported_counts.numel() >= count64, "supported_counts workspace is too small");
    TORCH_CHECK(keep_mask.numel() >= count64, "keep_mask workspace is too small");
    TORCH_CHECK(largest_component_size.numel() >= 1, "largest size workspace is empty");
    TORCH_CHECK(statistics.numel() >= 11, "statistics workspace must contain 11 values");
    TORCH_CHECK(tiny_component_max_voxels >= 0, "tiny threshold must be non-negative");
    TORCH_CHECK(tiny_component_max_voxels <= INT_MAX, "tiny threshold is too large");
    TORCH_CHECK(max_small_component_voxels >= 0, "small-component threshold must be non-negative");
    TORCH_CHECK(max_small_component_voxels <= INT_MAX, "small-component threshold is too large");
    TORCH_CHECK(max_small_component_voxels >= tiny_component_max_voxels,
                "small-component threshold must be at least the tiny threshold");
    TORCH_CHECK(support_radius_voxels >= 0, "support radius must be non-negative");
    TORCH_CHECK(support_radius_voxels <= INT_MAX, "support radius is too large");
    TORCH_CHECK(min_supported_fraction >= 0.0 && min_supported_fraction <= 1.0,
                "supported fraction must be in [0,1]");

    const int count = static_cast<int>(count64);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    voxel_component_filter_kernel_launcher(
        count,
        sorted_keys.data_ptr<int64_t>(),
        shifted_coords.data_ptr<int>(),
        absolute_coords.data_ptr<int>(),
        extents.data_ptr<int>(),
        previous_absolute_coords.data_ptr<int>(),
        static_cast<int>(previous_absolute_coords.size(0)),
        static_cast<int>(tiny_component_max_voxels),
        static_cast<int>(max_small_component_voxels),
        static_cast<int>(support_radius_voxels),
        static_cast<float>(min_supported_fraction),
        parents.data_ptr<int>(),
        component_sizes.data_ptr<int>(),
        supported_counts.data_ptr<int>(),
        largest_component_size.data_ptr<int>(),
        keep_mask.data_ptr<bool>(),
        statistics.data_ptr<int>(),
        stream);
}
