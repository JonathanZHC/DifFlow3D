#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ball_query_gpu.h"
#include "group_points_gpu.h"
#include "interpolate_gpu.h"
#include "sampling_gpu.h"

void gaussian_softmax_recovery_wrapper(
    at::Tensor queries,
    at::Tensor anchors,
    at::Tensor anchor_flow,
    double sigma,
    at::Tensor output_flow);

void gaussian_recovery_hash_build_wrapper(
    at::Tensor anchors,
    double cell_size,
    at::Tensor hash_heads,
    at::Tensor anchor_next,
    at::Tensor anchor_cells);

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
    at::Tensor local_neighbor_counts);

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
    at::Tensor local_neighbor_counts);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query_wrapper", &ball_query_wrapper_fast, "ball_query_wrapper_fast");

    m.def("group_points_wrapper", &group_points_wrapper_fast, "group_points_wrapper_fast");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_fast, "group_points_grad_wrapper_fast");

    m.def("gather_points_wrapper", &gather_points_wrapper_fast, "gather_points_wrapper_fast");
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast, "gather_points_grad_wrapper_fast");

    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");

    m.def("three_nn_wrapper", &three_nn_wrapper_fast, "three_nn_wrapper_fast");
    m.def("three_interpolate_wrapper", &three_interpolate_wrapper_fast, "three_interpolate_wrapper_fast");
    m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_fast, "three_interpolate_grad_wrapper_fast");

    m.def(
        "gaussian_softmax_recovery_wrapper",
        &gaussian_softmax_recovery_wrapper,
        "Exact global warp-per-query Gaussian-softmax recovery");
    m.def(
        "gaussian_recovery_hash_build_wrapper",
        &gaussian_recovery_hash_build_wrapper,
        "Build anchor hash grid for sparse local Gaussian recovery");
    m.def(
        "gaussian_softmax_recovery_local_wrapper",
        &gaussian_softmax_recovery_local_wrapper,
        "Radius-local hash-grid Gaussian-softmax recovery");
    m.def(
        "gaussian_softmax_recovery_local_track_aware_wrapper",
        &gaussian_softmax_recovery_local_track_aware_wrapper,
        "Track-aware radius-local hash-grid Gaussian-softmax recovery");
}
