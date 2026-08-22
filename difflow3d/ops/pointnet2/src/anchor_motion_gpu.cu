#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

#include "anchor_motion_gpu.h"

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int TRANSPORT_THREADS = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int ELEMENTWISE_THREADS = 256;

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ int warp_reduce_sum_int(int value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ uint32_t mix_cell_hash(int x, int y, int z) {
    uint32_t h = static_cast<uint32_t>(x) * 0x8da6b343u;
    h ^= static_cast<uint32_t>(y) * 0xd8163841u;
    h ^= static_cast<uint32_t>(z) * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ int hash_bucket(int x, int y, int z, int hash_size) {
    return static_cast<int>(
        mix_cell_hash(x, y, z) & static_cast<uint32_t>(hash_size - 1));
}

template <int C>
__device__ __forceinline__ void online_softmax_add_vector(
    float logit,
    const float *value,
    float &running_max,
    float &denominator,
    float (&weighted)[C]) {
    if (logit <= running_max) {
        const float weight = expf(logit - running_max);
        denominator += weight;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            weighted[c] += weight * value[c];
        }
    } else {
        const float old_scale = expf(running_max - logit);
        denominator = denominator * old_scale + 1.0f;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            weighted[c] = weighted[c] * old_scale + value[c];
        }
        running_max = logit;
    }
}

template <int C>
__device__ __forceinline__ void merge_and_write(
    int lane,
    int query_index,
    float local_max,
    float local_denominator,
    float (&weighted)[C],
    float *output_values) {
    float global_max = warp_reduce_max(local_max);
    global_max = __shfl_sync(0xffffffffu, global_max, 0);
    const float scale = expf(local_max - global_max);
    local_denominator *= scale;
#pragma unroll
    for (int c = 0; c < C; ++c) {
        weighted[c] *= scale;
    }

    const float denominator = warp_reduce_sum(local_denominator);
#pragma unroll
    for (int c = 0; c < C; ++c) {
        weighted[c] = warp_reduce_sum(weighted[c]);
    }

    if (lane == 0) {
        const float inv_denominator = 1.0f / fmaxf(denominator, 1.0e-20f);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            output_values[query_index * C + c] = weighted[c] * inv_denominator;
        }
    }
}

template <int C>
__device__ __forceinline__ int global_same_track_transport(
    int lane,
    int query_index,
    int anchor_count,
    int query_track_id,
    float inv_two_sigma2,
    float qx,
    float qy,
    float qz,
    const float *anchors,
    const float *anchor_values,
    const int *anchor_track_ids,
    float *output_values) {
    float local_max = -CUDART_INF_F;
    float denominator = 0.0f;
    float weighted[C] = {0.0f};
    int local_count = 0;

    for (int a = lane; a < anchor_count; a += WARP_SIZE) {
        if (anchor_track_ids[a] != query_track_id) {
            continue;
        }
        ++local_count;
        const float dx = qx - anchors[a * 3 + 0];
        const float dy = qy - anchors[a * 3 + 1];
        const float dz = qz - anchors[a * 3 + 2];
        const float logit = -(dx * dx + dy * dy + dz * dz) * inv_two_sigma2;
        online_softmax_add_vector<C>(
            logit,
            anchor_values + a * C,
            local_max,
            denominator,
            weighted);
    }

    const int total_reduced = warp_reduce_sum_int(local_count);
    const int total = __shfl_sync(0xffffffffu, total_reduced, 0);
    if (total > 0) {
        merge_and_write<C>(
            lane, query_index, local_max, denominator, weighted, output_values);
    } else if (lane == 0) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
            output_values[query_index * C + c] = 0.0f;
        }
    }
    return total;
}

template <int C>
__global__ void gaussian_softmax_transport_local_track_aware_kernel(
    int query_count,
    int anchor_count,
    int hash_size,
    float inv_two_sigma2,
    float radius2,
    float inv_cell_size,
    bool global_same_track_fallback,
    const float *__restrict__ queries,
    const float *__restrict__ anchors,
    const float *__restrict__ anchor_values,
    const int *__restrict__ query_track_ids,
    const int *__restrict__ anchor_track_ids,
    const int *__restrict__ hash_heads,
    const int *__restrict__ anchor_next,
    const int *__restrict__ anchor_cells,
    float *__restrict__ output_values,
    int *__restrict__ support_counts) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int q = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (q >= query_count) {
        return;
    }

    const float qx = queries[q * 3 + 0];
    const float qy = queries[q * 3 + 1];
    const float qz = queries[q * 3 + 2];
    const int query_track_id = query_track_ids[q];
    const int qcx = __float2int_rd(qx * inv_cell_size);
    const int qcy = __float2int_rd(qy * inv_cell_size);
    const int qcz = __float2int_rd(qz * inv_cell_size);

    float local_max = -CUDART_INF_F;
    float denominator = 0.0f;
    float weighted[C] = {0.0f};
    int local_count = 0;

    if (lane < 27) {
        const int dx_cell = lane % 3 - 1;
        const int dy_cell = (lane / 3) % 3 - 1;
        const int dz_cell = lane / 9 - 1;
        const int cx = qcx + dx_cell;
        const int cy = qcy + dy_cell;
        const int cz = qcz + dz_cell;
        const int bucket = hash_bucket(cx, cy, cz, hash_size);

        for (int a = hash_heads[bucket]; a >= 0; a = anchor_next[a]) {
            if (anchor_cells[a * 3 + 0] != cx ||
                anchor_cells[a * 3 + 1] != cy ||
                anchor_cells[a * 3 + 2] != cz ||
                anchor_track_ids[a] != query_track_id) {
                continue;
            }
            const float dx = qx - anchors[a * 3 + 0];
            const float dy = qy - anchors[a * 3 + 1];
            const float dz = qz - anchors[a * 3 + 2];
            const float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 > radius2) {
                continue;
            }
            ++local_count;
            const float logit = -dist2 * inv_two_sigma2;
            online_softmax_add_vector<C>(
                logit,
                anchor_values + a * C,
                local_max,
                denominator,
                weighted);
        }
    }

    const int total_reduced = warp_reduce_sum_int(local_count);
    const int total = __shfl_sync(0xffffffffu, total_reduced, 0);
    if (total > 0) {
        if (lane == 0) {
            support_counts[q] = total;
        }
        merge_and_write<C>(
            lane, q, local_max, denominator, weighted, output_values);
        return;
    }

    if (global_same_track_fallback) {
        const int global_count = global_same_track_transport<C>(
            lane,
            q,
            anchor_count,
            query_track_id,
            inv_two_sigma2,
            qx,
            qy,
            qz,
            anchors,
            anchor_values,
            anchor_track_ids,
            output_values);
        if (lane == 0) {
            support_counts[q] = global_count;
        }
        return;
    }

    if (lane == 0) {
        support_counts[q] = 0;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            output_values[q * C + c] = 0.0f;
        }
    }
}

__device__ __forceinline__ bool finite3(float x, float y, float z) {
    return isfinite(x) && isfinite(y) && isfinite(z);
}

__device__ __forceinline__ float clamp_velocity_variance(float variance) {
    constexpr float kMinVariance = 1.0e-12f;
    constexpr float kMaxVariance = 1.0e12f;
    if (!isfinite(variance)) {
        return 1.0f;
    }
    return fminf(fmaxf(variance, kMinVariance), kMaxVariance);
}

__device__ __forceinline__ void write_initialized_velocity_kf_state(
    int i,
    float zx,
    float zy,
    float zz,
    bool measurement_finite,
    float initial_velocity_variance,
    float *output_state6) {
    output_state6[i * 6 + 0] = measurement_finite ? zx : 0.0f;
    output_state6[i * 6 + 1] = measurement_finite ? zy : 0.0f;
    output_state6[i * 6 + 2] = measurement_finite ? zz : 0.0f;
    output_state6[i * 6 + 3] = initial_velocity_variance;
    output_state6[i * 6 + 4] = initial_velocity_variance;
    output_state6[i * 6 + 5] = initial_velocity_variance;
}

__device__ __forceinline__ void write_predicted_velocity_kf_state(
    int i,
    float vx,
    float vy,
    float vz,
    float pvx,
    float pvy,
    float pvz,
    float *output_state6) {
    output_state6[i * 6 + 0] = vx;
    output_state6[i * 6 + 1] = vy;
    output_state6[i * 6 + 2] = vz;
    output_state6[i * 6 + 3] = pvx;
    output_state6[i * 6 + 4] = pvy;
    output_state6[i * 6 + 5] = pvz;
}

// Velocity-only random-walk Kalman filter.
// State per anchor: [vx, vy, vz, Pvx, Pvy, Pvz]. There is deliberately no
// acceleration state and no statistical innovation gate. Q_v is passed
// directly as a per-update velocity variance.
__global__ void anchor_kalman_update_kernel(
    int count,
    float dt,
    float process_velocity_variance,
    float measurement_variance,
    float initial_velocity_variance,
    float min_innovation_variance,
    const float *__restrict__ current_flow,
    const float *__restrict__ transported_previous_state6,
    const int *__restrict__ support_counts,
    float *__restrict__ output_state6) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const float inv_dt = 1.0f / dt;
    const float zx = current_flow[i * 3 + 0] * inv_dt;
    const float zy = current_flow[i * 3 + 1] * inv_dt;
    const float zz = current_flow[i * 3 + 2] * inv_dt;
    const bool measurement_finite = finite3(zx, zy, zz);

    // No transported history: initialize from the current finite measurement.
    if (support_counts[i] <= 0) {
        write_initialized_velocity_kf_state(
            i,
            zx,
            zy,
            zz,
            measurement_finite,
            initial_velocity_variance,
            output_state6);
        return;
    }

    const float prev_vx = transported_previous_state6[i * 6 + 0];
    const float prev_vy = transported_previous_state6[i * 6 + 1];
    const float prev_vz = transported_previous_state6[i * 6 + 2];
    const float prev_pvx = transported_previous_state6[i * 6 + 3];
    const float prev_pvy = transported_previous_state6[i * 6 + 4];
    const float prev_pvz = transported_previous_state6[i * 6 + 5];

    const bool history_finite =
        finite3(prev_vx, prev_vy, prev_vz) &&
        isfinite(prev_pvx) && isfinite(prev_pvy) && isfinite(prev_pvz);
    if (!history_finite) {
        write_initialized_velocity_kf_state(
            i,
            zx,
            zy,
            zz,
            measurement_finite,
            initial_velocity_variance,
            output_state6);
        return;
    }

    // Random-walk prediction: v_k^- = v_{k-1}, P_k^- = P_{k-1} + Q_v.
    const float pred_vx = prev_vx;
    const float pred_vy = prev_vy;
    const float pred_vz = prev_vz;
    const float pred_pvx = clamp_velocity_variance(prev_pvx + process_velocity_variance);
    const float pred_pvy = clamp_velocity_variance(prev_pvy + process_velocity_variance);
    const float pred_pvz = clamp_velocity_variance(prev_pvz + process_velocity_variance);

    // A non-finite sensor/model output cannot participate in the arithmetic.
    // Keep the finite random-walk prediction without adding any statistical
    // statistical measurement gating.
    if (!measurement_finite) {
        write_predicted_velocity_kf_state(
            i,
            pred_vx,
            pred_vy,
            pred_vz,
            pred_pvx,
            pred_pvy,
            pred_pvz,
            output_state6);
        return;
    }

    const float rx = zx - pred_vx;
    const float ry = zy - pred_vy;
    const float rz = zz - pred_vz;
    const float sx = fmaxf(pred_pvx + measurement_variance, min_innovation_variance);
    const float sy = fmaxf(pred_pvy + measurement_variance, min_innovation_variance);
    const float sz = fmaxf(pred_pvz + measurement_variance, min_innovation_variance);

    const float kx = pred_pvx / sx;
    const float ky = pred_pvy / sy;
    const float kz = pred_pvz / sz;
    const float updated_vx = pred_vx + kx * rx;
    const float updated_vy = pred_vy + ky * ry;
    const float updated_vz = pred_vz + kz * rz;

    // Joseph-form scalar covariance update for each independent velocity axis.
    const float one_minus_kx = 1.0f - kx;
    const float one_minus_ky = 1.0f - ky;
    const float one_minus_kz = 1.0f - kz;
    float updated_pvx = one_minus_kx * one_minus_kx * pred_pvx
                      + kx * kx * measurement_variance;
    float updated_pvy = one_minus_ky * one_minus_ky * pred_pvy
                      + ky * ky * measurement_variance;
    float updated_pvz = one_minus_kz * one_minus_kz * pred_pvz
                      + kz * kz * measurement_variance;
    updated_pvx = clamp_velocity_variance(updated_pvx);
    updated_pvy = clamp_velocity_variance(updated_pvy);
    updated_pvz = clamp_velocity_variance(updated_pvz);

    // Numerical-safety fallback only. With finite input and positive variances
    // this branch should not normally be taken.
    if (!finite3(updated_vx, updated_vy, updated_vz) ||
        !isfinite(updated_pvx) || !isfinite(updated_pvy) || !isfinite(updated_pvz)) {
        write_predicted_velocity_kf_state(
            i,
            pred_vx,
            pred_vy,
            pred_vz,
            pred_pvx,
            pred_pvy,
            pred_pvz,
            output_state6);
        return;
    }

    output_state6[i * 6 + 0] = updated_vx;
    output_state6[i * 6 + 1] = updated_vy;
    output_state6[i * 6 + 2] = updated_vz;
    output_state6[i * 6 + 3] = updated_pvx;
    output_state6[i * 6 + 4] = updated_pvy;
    output_state6[i * 6 + 5] = updated_pvz;
}

template <int C>
void launch_transport(
    int query_count,
    int anchor_count,
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
    cudaStream_t stream) {
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);
    const float radius2 = radius * radius;
    const float inv_cell_size = 1.0f / cell_size;
    gaussian_softmax_transport_local_track_aware_kernel<C>
        <<<blocks, TRANSPORT_THREADS, 0, stream>>>(
            query_count,
            anchor_count,
            hash_size,
            inv_two_sigma2,
            radius2,
            inv_cell_size,
            global_same_track_fallback,
            queries,
            anchors,
            anchor_values,
            query_track_ids,
            anchor_track_ids,
            hash_heads,
            anchor_next,
            anchor_cells,
            output_values,
            support_counts);
}

}  // namespace

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
    cudaStream_t stream) {
    switch (channels) {
        case 3:
            launch_transport<3>(query_count, anchor_count, hash_size, sigma, radius,
                                cell_size, global_same_track_fallback, queries, anchors,
                                anchor_values, query_track_ids, anchor_track_ids,
                                hash_heads, anchor_next, anchor_cells, output_values,
                                support_counts, stream);
            break;
        case 6:
            launch_transport<6>(query_count, anchor_count, hash_size, sigma, radius,
                                cell_size, global_same_track_fallback, queries, anchors,
                                anchor_values, query_track_ids, anchor_track_ids,
                                hash_heads, anchor_next, anchor_cells, output_values,
                                support_counts, stream);
            break;
    }
}

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
    cudaStream_t stream) {
    const int blocks = (count + ELEMENTWISE_THREADS - 1) / ELEMENTWISE_THREADS;
    anchor_kalman_update_kernel<<<blocks, ELEMENTWISE_THREADS, 0, stream>>>(
        count,
        dt,
        process_velocity_variance,
        measurement_variance,
        initial_velocity_variance,
        min_innovation_variance,
        current_flow,
        transported_previous_state6,
        support_counts,
        output_state6);
}
