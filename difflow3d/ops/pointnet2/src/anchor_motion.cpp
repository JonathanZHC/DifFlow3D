#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "anchor_motion_gpu.h"

namespace {

void require_cuda_float_matrix(
    const at::Tensor &tensor,
    const char *name,
    int64_t columns) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == columns,
                name, " must be [N,", columns, "]");
}

void require_cuda_int_vector(
    const at::Tensor &tensor,
    const char *name,
    int64_t count) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == count,
                name, " must be [", count, "]");
}

void require_same_device(
    const at::Tensor &reference,
    const at::Tensor &tensor,
    const char *name) {
    TORCH_CHECK(reference.device() == tensor.device(),
                name, " must be on the same CUDA device");
}

}  // namespace

void gaussian_softmax_transport_local_track_aware_wrapper(
    at::Tensor queries,
    at::Tensor anchors,
    at::Tensor anchor_values,
    at::Tensor query_track_ids,
    at::Tensor anchor_track_ids,
    double sigma,
    double radius,
    double cell_size,
    bool global_same_track_fallback,
    at::Tensor hash_heads,
    at::Tensor anchor_next,
    at::Tensor anchor_cells,
    at::Tensor output_values,
    at::Tensor support_counts) {
    require_cuda_float_matrix(queries, "queries", 3);
    require_cuda_float_matrix(anchors, "anchors", 3);
    TORCH_CHECK(anchor_values.is_cuda() && anchor_values.scalar_type() == at::kFloat
                && anchor_values.is_contiguous() && anchor_values.dim() == 2,
                "anchor_values must be contiguous CUDA float32 [K,C]");
    const int64_t channels = anchor_values.size(1);
    TORCH_CHECK(channels == 3 || channels == 6,
                "anchor_values channels must be 3 or 6");
    TORCH_CHECK(anchor_values.size(0) == anchors.size(0),
                "anchor_values first dimension must match anchors");
    TORCH_CHECK(queries.size(0) > 0 && anchors.size(0) > 0,
                "queries and anchors must be non-empty");

    require_cuda_int_vector(query_track_ids, "query_track_ids", queries.size(0));
    require_cuda_int_vector(anchor_track_ids, "anchor_track_ids", anchors.size(0));
    require_cuda_int_vector(support_counts, "support_counts", queries.size(0));

    TORCH_CHECK(output_values.is_cuda() && output_values.scalar_type() == at::kFloat
                && output_values.is_contiguous() && output_values.dim() == 2
                && output_values.size(0) == queries.size(0)
                && output_values.size(1) == channels,
                "output_values must be contiguous CUDA float32 [Q,C]");

    require_cuda_int_vector(hash_heads, "hash_heads", hash_heads.size(0));
    require_cuda_int_vector(anchor_next, "anchor_next", anchors.size(0));
    TORCH_CHECK(anchor_cells.is_cuda() && anchor_cells.scalar_type() == at::kInt
                && anchor_cells.is_contiguous() && anchor_cells.dim() == 2
                && anchor_cells.size(0) == anchors.size(0) && anchor_cells.size(1) == 3,
                "anchor_cells must be contiguous CUDA int32 [K,3]");
    TORCH_CHECK(hash_heads.size(0) > 0
                && (hash_heads.size(0) & (hash_heads.size(0) - 1)) == 0,
                "hash_heads size must be a positive power of two");

    require_same_device(queries, anchors, "anchors");
    require_same_device(queries, anchor_values, "anchor_values");
    require_same_device(queries, query_track_ids, "query_track_ids");
    require_same_device(queries, anchor_track_ids, "anchor_track_ids");
    require_same_device(queries, hash_heads, "hash_heads");
    require_same_device(queries, anchor_next, "anchor_next");
    require_same_device(queries, anchor_cells, "anchor_cells");
    require_same_device(queries, output_values, "output_values");
    require_same_device(queries, support_counts, "support_counts");

    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
    TORCH_CHECK(radius > 0.0, "radius must be positive");
    TORCH_CHECK(cell_size >= radius,
                "cell_size must be >= radius so 27 neighboring cells cover the cutoff");

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    gaussian_softmax_transport_local_track_aware_kernel_launcher(
        static_cast<int>(queries.size(0)),
        static_cast<int>(anchors.size(0)),
        static_cast<int>(channels),
        static_cast<int>(hash_heads.size(0)),
        static_cast<float>(sigma),
        static_cast<float>(radius),
        static_cast<float>(cell_size),
        global_same_track_fallback,
        queries.data_ptr<float>(),
        anchors.data_ptr<float>(),
        anchor_values.data_ptr<float>(),
        query_track_ids.data_ptr<int>(),
        anchor_track_ids.data_ptr<int>(),
        hash_heads.data_ptr<int>(),
        anchor_next.data_ptr<int>(),
        anchor_cells.data_ptr<int>(),
        output_values.data_ptr<float>(),
        support_counts.data_ptr<int>(),
        stream);
}

void anchor_kalman_update_wrapper(
    at::Tensor current_flow,
    at::Tensor transported_previous_state6,
    at::Tensor support_counts,
    double dt,
    double process_velocity_variance,
    double measurement_variance,
    double initial_velocity_variance,
    double min_innovation_variance,
    at::Tensor output_state6) {
    require_cuda_float_matrix(current_flow, "current_flow", 3);
    require_cuda_float_matrix(transported_previous_state6, "transported_previous_state6", 6);
    require_cuda_int_vector(support_counts, "support_counts", current_flow.size(0));
    require_cuda_float_matrix(output_state6, "output_state6", 6);
    TORCH_CHECK(transported_previous_state6.size(0) == current_flow.size(0)
                && output_state6.size(0) == current_flow.size(0),
                "KF tensors must have the same anchor count");

    require_same_device(current_flow, transported_previous_state6, "transported_previous_state6");
    require_same_device(current_flow, support_counts, "support_counts");
    require_same_device(current_flow, output_state6, "output_state6");

    TORCH_CHECK(dt > 0.0, "dt must be positive");
    TORCH_CHECK(process_velocity_variance >= 0.0,
                "process_velocity_variance must be non-negative");
    TORCH_CHECK(measurement_variance > 0.0, "measurement_variance must be positive");
    TORCH_CHECK(initial_velocity_variance > 0.0, "initial_velocity_variance must be positive");
    TORCH_CHECK(min_innovation_variance > 0.0, "min_innovation_variance must be positive");

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    anchor_kalman_update_kernel_launcher(
        static_cast<int>(current_flow.size(0)),
        static_cast<float>(dt),
        static_cast<float>(process_velocity_variance),
        static_cast<float>(measurement_variance),
        static_cast<float>(initial_velocity_variance),
        static_cast<float>(min_innovation_variance),
        current_flow.data_ptr<float>(),
        transported_previous_state6.data_ptr<float>(),
        support_counts.data_ptr<int>(),
        output_state6.data_ptr<float>(),
        stream);
}
