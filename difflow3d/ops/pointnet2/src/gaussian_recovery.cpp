#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "gaussian_recovery_gpu.h"

namespace {

void validate_common(
    const at::Tensor &queries,
    const at::Tensor &anchors,
    const at::Tensor &anchor_flow,
    const at::Tensor &output_flow,
    double sigma) {
    TORCH_CHECK(
        queries.is_cuda() && anchors.is_cuda() && anchor_flow.is_cuda() && output_flow.is_cuda(),
        "Gaussian recovery tensors must be CUDA");
    TORCH_CHECK(
        queries.device() == anchors.device() &&
        queries.device() == anchor_flow.device() &&
        queries.device() == output_flow.device(),
        "Gaussian recovery tensors must be on the same CUDA device");
    TORCH_CHECK(
        queries.scalar_type() == at::kFloat && anchors.scalar_type() == at::kFloat &&
        anchor_flow.scalar_type() == at::kFloat && output_flow.scalar_type() == at::kFloat,
        "Gaussian recovery tensors must be float32");
    TORCH_CHECK(
        queries.is_contiguous() && anchors.is_contiguous() &&
        anchor_flow.is_contiguous() && output_flow.is_contiguous(),
        "Gaussian recovery tensors must be contiguous");
    TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 3, "queries must be [Q,3]");
    TORCH_CHECK(anchors.dim() == 2 && anchors.size(1) == 3, "anchors must be [K,3]");
    TORCH_CHECK(queries.size(0) > 0, "queries must be non-empty");
    TORCH_CHECK(anchors.size(0) > 0, "anchors must be non-empty");
    TORCH_CHECK(anchor_flow.sizes() == anchors.sizes(), "anchor_flow must match anchors");
    TORCH_CHECK(output_flow.sizes() == queries.sizes(), "output_flow must match queries");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
}

}  // namespace

void gaussian_softmax_recovery_wrapper(
    at::Tensor queries,
    at::Tensor anchors,
    at::Tensor anchor_flow,
    double sigma,
    at::Tensor output_flow) {
    validate_common(queries, anchors, anchor_flow, output_flow, sigma);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    gaussian_softmax_recovery_kernel_launcher(
        static_cast<int>(queries.size(0)),
        static_cast<int>(anchors.size(0)),
        static_cast<float>(sigma),
        queries.data_ptr<float>(),
        anchors.data_ptr<float>(),
        anchor_flow.data_ptr<float>(),
        output_flow.data_ptr<float>(),
        stream);
}

void gaussian_recovery_hash_build_wrapper(
    at::Tensor anchors,
    double cell_size,
    at::Tensor hash_heads,
    at::Tensor anchor_next,
    at::Tensor anchor_cells) {
    TORCH_CHECK(anchors.is_cuda() && hash_heads.is_cuda() && anchor_next.is_cuda() && anchor_cells.is_cuda(),
                "Gaussian hash tensors must be CUDA");
    TORCH_CHECK(anchors.device() == hash_heads.device()
                && anchors.device() == anchor_next.device()
                && anchors.device() == anchor_cells.device(),
                "Gaussian hash tensors must be on the same CUDA device");
    TORCH_CHECK(anchors.scalar_type() == at::kFloat, "anchors must be float32");
    TORCH_CHECK(hash_heads.scalar_type() == at::kInt
                && anchor_next.scalar_type() == at::kInt
                && anchor_cells.scalar_type() == at::kInt,
                "Gaussian hash index tensors must be int32");
    TORCH_CHECK(anchors.is_contiguous() && hash_heads.is_contiguous()
                && anchor_next.is_contiguous() && anchor_cells.is_contiguous(),
                "Gaussian hash tensors must be contiguous");
    TORCH_CHECK(anchors.dim() == 2 && anchors.size(1) == 3, "anchors must be [K,3]");
    TORCH_CHECK(anchors.size(0) > 0, "anchors must be non-empty");
    TORCH_CHECK(hash_heads.dim() == 1 && hash_heads.size(0) > 0, "hash_heads must be [H]");
    TORCH_CHECK((hash_heads.size(0) & (hash_heads.size(0) - 1)) == 0,
                "hash_heads size must be a power of two");
    TORCH_CHECK(anchor_next.dim() == 1 && anchor_next.size(0) == anchors.size(0),
                "anchor_next must be [K]");
    TORCH_CHECK(anchor_cells.dim() == 2 && anchor_cells.size(0) == anchors.size(0)
                && anchor_cells.size(1) == 3, "anchor_cells must be [K,3]");
    TORCH_CHECK(cell_size > 0.0, "cell_size must be positive");

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    // -1 is all bits set for int32. Clearing here keeps hash-grid setup on the
    // same CUDA stream and avoids a separate Python torch.full kernel.
    cudaMemsetAsync(
        hash_heads.data_ptr<int>(),
        0xFF,
        static_cast<size_t>(hash_heads.size(0)) * sizeof(int),
        stream);
    gaussian_recovery_hash_build_kernel_launcher(
        static_cast<int>(anchors.size(0)),
        static_cast<int>(hash_heads.size(0)),
        static_cast<float>(cell_size),
        anchors.data_ptr<float>(),
        hash_heads.data_ptr<int>(),
        anchor_next.data_ptr<int>(),
        anchor_cells.data_ptr<int>(),
        stream);
}

void gaussian_softmax_recovery_local_wrapper(
    at::Tensor queries,
    at::Tensor anchors,
    at::Tensor anchor_flow,
    double sigma,
    double radius,
    double cell_size,
    at::Tensor hash_heads,
    at::Tensor anchor_next,
    at::Tensor anchor_cells,
    at::Tensor output_flow,
    at::Tensor local_neighbor_counts) {
    validate_common(queries, anchors, anchor_flow, output_flow, sigma);
    TORCH_CHECK(radius > 0.0, "radius must be positive");
    TORCH_CHECK(cell_size >= radius,
                "local recovery requires cell_size >= radius so 27 neighboring cells cover the cutoff");
    TORCH_CHECK(hash_heads.is_cuda() && anchor_next.is_cuda() && anchor_cells.is_cuda()
                && local_neighbor_counts.is_cuda(), "Local recovery hash tensors must be CUDA");
    TORCH_CHECK(queries.device() == hash_heads.device()
                && queries.device() == anchor_next.device()
                && queries.device() == anchor_cells.device()
                && queries.device() == local_neighbor_counts.device(),
                "Local recovery tensors must be on the same CUDA device");
    TORCH_CHECK(hash_heads.scalar_type() == at::kInt
                && anchor_next.scalar_type() == at::kInt
                && anchor_cells.scalar_type() == at::kInt
                && local_neighbor_counts.scalar_type() == at::kInt,
                "Local recovery hash/count tensors must be int32");
    TORCH_CHECK(hash_heads.is_contiguous() && anchor_next.is_contiguous()
                && anchor_cells.is_contiguous() && local_neighbor_counts.is_contiguous(),
                "Local recovery hash/count tensors must be contiguous");
    TORCH_CHECK(hash_heads.dim() == 1 && hash_heads.size(0) > 0, "hash_heads must be [H]");
    TORCH_CHECK((hash_heads.size(0) & (hash_heads.size(0) - 1)) == 0,
                "hash_heads size must be a power of two");
    TORCH_CHECK(anchor_next.dim() == 1 && anchor_next.size(0) == anchors.size(0),
                "anchor_next must be [K]");
    TORCH_CHECK(anchor_cells.dim() == 2 && anchor_cells.size(0) == anchors.size(0)
                && anchor_cells.size(1) == 3, "anchor_cells must be [K,3]");
    TORCH_CHECK(local_neighbor_counts.dim() == 1
                && local_neighbor_counts.size(0) == queries.size(0),
                "local_neighbor_counts must be [Q]");

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    gaussian_softmax_recovery_local_kernel_launcher(
        static_cast<int>(queries.size(0)),
        static_cast<int>(anchors.size(0)),
        static_cast<int>(hash_heads.size(0)),
        static_cast<float>(sigma),
        static_cast<float>(radius),
        static_cast<float>(cell_size),
        queries.data_ptr<float>(),
        anchors.data_ptr<float>(),
        anchor_flow.data_ptr<float>(),
        hash_heads.data_ptr<int>(),
        anchor_next.data_ptr<int>(),
        anchor_cells.data_ptr<int>(),
        output_flow.data_ptr<float>(),
        local_neighbor_counts.data_ptr<int>(),
        stream);
}

void gaussian_softmax_recovery_local_track_aware_wrapper(
    at::Tensor queries,
    at::Tensor anchors,
    at::Tensor anchor_flow,
    at::Tensor query_track_ids,
    at::Tensor anchor_track_ids,
    double sigma,
    double radius,
    double cell_size,
    at::Tensor hash_heads,
    at::Tensor anchor_next,
    at::Tensor anchor_cells,
    at::Tensor output_flow,
    at::Tensor local_neighbor_counts) {
    validate_common(queries, anchors, anchor_flow, output_flow, sigma);
    TORCH_CHECK(radius > 0.0, "radius must be positive");
    TORCH_CHECK(cell_size >= radius,
                "local recovery requires cell_size >= radius so 27 neighboring cells cover the cutoff");
    TORCH_CHECK(query_track_ids.is_cuda() && anchor_track_ids.is_cuda(),
                "Track ID tensors must be CUDA");
    TORCH_CHECK(query_track_ids.device() == queries.device()
                && anchor_track_ids.device() == queries.device(),
                "Track ID tensors must be on the same CUDA device as recovery tensors");
    TORCH_CHECK(query_track_ids.scalar_type() == at::kInt
                && anchor_track_ids.scalar_type() == at::kInt,
                "Track ID tensors must be int32");
    TORCH_CHECK(query_track_ids.is_contiguous() && anchor_track_ids.is_contiguous(),
                "Track ID tensors must be contiguous");
    TORCH_CHECK(query_track_ids.dim() == 1
                && query_track_ids.size(0) == queries.size(0),
                "query_track_ids must be [Q]");
    TORCH_CHECK(anchor_track_ids.dim() == 1
                && anchor_track_ids.size(0) == anchors.size(0),
                "anchor_track_ids must be [K]");

    TORCH_CHECK(hash_heads.is_cuda() && anchor_next.is_cuda() && anchor_cells.is_cuda()
                && local_neighbor_counts.is_cuda(), "Local recovery hash tensors must be CUDA");
    TORCH_CHECK(queries.device() == hash_heads.device()
                && queries.device() == anchor_next.device()
                && queries.device() == anchor_cells.device()
                && queries.device() == local_neighbor_counts.device(),
                "Local recovery tensors must be on the same CUDA device");
    TORCH_CHECK(hash_heads.scalar_type() == at::kInt
                && anchor_next.scalar_type() == at::kInt
                && anchor_cells.scalar_type() == at::kInt
                && local_neighbor_counts.scalar_type() == at::kInt,
                "Local recovery hash/count tensors must be int32");
    TORCH_CHECK(hash_heads.is_contiguous() && anchor_next.is_contiguous()
                && anchor_cells.is_contiguous() && local_neighbor_counts.is_contiguous(),
                "Local recovery hash/count tensors must be contiguous");
    TORCH_CHECK(hash_heads.dim() == 1 && hash_heads.size(0) > 0, "hash_heads must be [H]");
    TORCH_CHECK((hash_heads.size(0) & (hash_heads.size(0) - 1)) == 0,
                "hash_heads size must be a power of two");
    TORCH_CHECK(anchor_next.dim() == 1 && anchor_next.size(0) == anchors.size(0),
                "anchor_next must be [K]");
    TORCH_CHECK(anchor_cells.dim() == 2 && anchor_cells.size(0) == anchors.size(0)
                && anchor_cells.size(1) == 3, "anchor_cells must be [K,3]");
    TORCH_CHECK(local_neighbor_counts.dim() == 1
                && local_neighbor_counts.size(0) == queries.size(0),
                "local_neighbor_counts must be [Q]");

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    gaussian_softmax_recovery_local_track_aware_kernel_launcher(
        static_cast<int>(queries.size(0)),
        static_cast<int>(anchors.size(0)),
        static_cast<int>(hash_heads.size(0)),
        static_cast<float>(sigma),
        static_cast<float>(radius),
        static_cast<float>(cell_size),
        queries.data_ptr<float>(),
        anchors.data_ptr<float>(),
        anchor_flow.data_ptr<float>(),
        query_track_ids.data_ptr<int>(),
        anchor_track_ids.data_ptr<int>(),
        hash_heads.data_ptr<int>(),
        anchor_next.data_ptr<int>(),
        anchor_cells.data_ptr<int>(),
        output_flow.data_ptr<float>(),
        local_neighbor_counts.data_ptr<int>(),
        stream);
}

