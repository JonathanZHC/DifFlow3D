#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

#include "gaussian_recovery_gpu.h"

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int RECOVERY_THREADS = WARP_SIZE * WARPS_PER_BLOCK;

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

__device__ __forceinline__ void online_softmax_add(
    float logit,
    float flow_x,
    float flow_y,
    float flow_z,
    float &running_max,
    float &denominator,
    float &weighted_x,
    float &weighted_y,
    float &weighted_z) {
    if (logit <= running_max) {
        const float weight = expf(logit - running_max);
        denominator += weight;
        weighted_x += weight * flow_x;
        weighted_y += weight * flow_y;
        weighted_z += weight * flow_z;
    } else {
        const float old_scale = expf(running_max - logit);
        denominator = denominator * old_scale + 1.0f;
        weighted_x = weighted_x * old_scale + flow_x;
        weighted_y = weighted_y * old_scale + flow_y;
        weighted_z = weighted_z * old_scale + flow_z;
        running_max = logit;
    }
}

__device__ __forceinline__ void merge_and_write_warp_softmax(
    int lane,
    int q,
    float local_max,
    float local_denominator,
    float local_flow_x,
    float local_flow_y,
    float local_flow_z,
    float *output_flow) {
    float global_max = warp_reduce_max(local_max);
    global_max = __shfl_sync(0xffffffffu, global_max, 0);

    const float rescale = expf(local_max - global_max);
    local_denominator *= rescale;
    local_flow_x *= rescale;
    local_flow_y *= rescale;
    local_flow_z *= rescale;

    const float denominator = warp_reduce_sum(local_denominator);
    const float flow_x = warp_reduce_sum(local_flow_x);
    const float flow_y = warp_reduce_sum(local_flow_y);
    const float flow_z = warp_reduce_sum(local_flow_z);

    if (lane == 0) {
        const float inv_denominator = 1.0f / fmaxf(denominator, 1.0e-20f);
        output_flow[q * 3 + 0] = flow_x * inv_denominator;
        output_flow[q * 3 + 1] = flow_y * inv_denominator;
        output_flow[q * 3 + 2] = flow_z * inv_denominator;
    }
}

// ---------------------------------------------------------------------------
// Exact global warp-per-query baseline.
// ---------------------------------------------------------------------------

__global__ void gaussian_softmax_recovery_warp_kernel(
    int query_count,
    int anchor_count,
    float inv_two_sigma2,
    const float *__restrict__ queries,
    const float *__restrict__ anchors,
    const float *__restrict__ anchor_flow,
    float *__restrict__ output_flow) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int q = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (q >= query_count) {
        return;
    }

    const float qx = queries[q * 3 + 0];
    const float qy = queries[q * 3 + 1];
    const float qz = queries[q * 3 + 2];

    float local_max = -CUDART_INF_F;
    float local_denominator = 0.0f;
    float local_flow_x = 0.0f;
    float local_flow_y = 0.0f;
    float local_flow_z = 0.0f;

    for (int a = lane; a < anchor_count; a += WARP_SIZE) {
        const float ax = anchors[a * 3 + 0];
        const float ay = anchors[a * 3 + 1];
        const float az = anchors[a * 3 + 2];
        const float dx = qx - ax;
        const float dy = qy - ay;
        const float dz = qz - az;
        const float logit = -(dx * dx + dy * dy + dz * dz) * inv_two_sigma2;
        online_softmax_add(
            logit,
            anchor_flow[a * 3 + 0],
            anchor_flow[a * 3 + 1],
            anchor_flow[a * 3 + 2],
            local_max,
            local_denominator,
            local_flow_x,
            local_flow_y,
            local_flow_z);
    }

    merge_and_write_warp_softmax(
        lane,
        q,
        local_max,
        local_denominator,
        local_flow_x,
        local_flow_y,
        local_flow_z,
        output_flow);
}

// ---------------------------------------------------------------------------
// Sparse local Gaussian backend.
// ---------------------------------------------------------------------------

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

__device__ __forceinline__ int hash_bucket(
    int x,
    int y,
    int z,
    int hash_size) {
    // Python allocates a power-of-two table, so masking is cheaper than modulo.
    return static_cast<int>(mix_cell_hash(x, y, z) & static_cast<uint32_t>(hash_size - 1));
}

__global__ void gaussian_recovery_hash_build_kernel(
    int anchor_count,
    int hash_size,
    float inv_cell_size,
    const float *__restrict__ anchors,
    int *__restrict__ hash_heads,
    int *__restrict__ anchor_next,
    int *__restrict__ anchor_cells) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= anchor_count) {
        return;
    }
    const int cx = __float2int_rd(anchors[a * 3 + 0] * inv_cell_size);
    const int cy = __float2int_rd(anchors[a * 3 + 1] * inv_cell_size);
    const int cz = __float2int_rd(anchors[a * 3 + 2] * inv_cell_size);
    anchor_cells[a * 3 + 0] = cx;
    anchor_cells[a * 3 + 1] = cy;
    anchor_cells[a * 3 + 2] = cz;
    const int bucket = hash_bucket(cx, cy, cz, hash_size);
    anchor_next[a] = atomicExch(&hash_heads[bucket], a);
}

__device__ __forceinline__ void warp_global_softmax_fallback(
    int lane,
    int q,
    int anchor_count,
    float inv_two_sigma2,
    const float qx,
    const float qy,
    const float qz,
    const float *__restrict__ anchors,
    const float *__restrict__ anchor_flow,
    float *__restrict__ output_flow) {
    float local_max = -CUDART_INF_F;
    float local_denominator = 0.0f;
    float local_flow_x = 0.0f;
    float local_flow_y = 0.0f;
    float local_flow_z = 0.0f;
    for (int a = lane; a < anchor_count; a += WARP_SIZE) {
        const float dx = qx - anchors[a * 3 + 0];
        const float dy = qy - anchors[a * 3 + 1];
        const float dz = qz - anchors[a * 3 + 2];
        const float logit = -(dx * dx + dy * dy + dz * dz) * inv_two_sigma2;
        online_softmax_add(
            logit,
            anchor_flow[a * 3 + 0],
            anchor_flow[a * 3 + 1],
            anchor_flow[a * 3 + 2],
            local_max,
            local_denominator,
            local_flow_x,
            local_flow_y,
            local_flow_z);
    }
    merge_and_write_warp_softmax(
        lane,
        q,
        local_max,
        local_denominator,
        local_flow_x,
        local_flow_y,
        local_flow_z,
        output_flow);
}

// Cell size equals the radius cutoff. Any anchor within the Euclidean cutoff
// must therefore lie in one of the 27 cells around the query cell. Lanes 0..26
// each own one neighboring cell and traverse only that hash bucket. Hash
// collisions are rejected by checking the stored integer cell coordinates.
__global__ void gaussian_softmax_recovery_local_kernel(
    int query_count,
    int anchor_count,
    int hash_size,
    float inv_two_sigma2,
    float radius2,
    float inv_cell_size,
    const float *__restrict__ queries,
    const float *__restrict__ anchors,
    const float *__restrict__ anchor_flow,
    const int *__restrict__ hash_heads,
    const int *__restrict__ anchor_next,
    const int *__restrict__ anchor_cells,
    float *__restrict__ output_flow,
    int *__restrict__ local_neighbor_counts) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int q = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (q >= query_count) {
        return;
    }

    const float qx = queries[q * 3 + 0];
    const float qy = queries[q * 3 + 1];
    const float qz = queries[q * 3 + 2];
    const int qcx = __float2int_rd(qx * inv_cell_size);
    const int qcy = __float2int_rd(qy * inv_cell_size);
    const int qcz = __float2int_rd(qz * inv_cell_size);

    float local_max = -CUDART_INF_F;
    float local_denominator = 0.0f;
    float local_flow_x = 0.0f;
    float local_flow_y = 0.0f;
    float local_flow_z = 0.0f;
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
                anchor_cells[a * 3 + 2] != cz) {
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
            online_softmax_add(
                logit,
                anchor_flow[a * 3 + 0],
                anchor_flow[a * 3 + 1],
                anchor_flow[a * 3 + 2],
                local_max,
                local_denominator,
                local_flow_x,
                local_flow_y,
                local_flow_z);
        }
    }

    const int total_count_reduced = warp_reduce_sum_int(local_count);
    const int total_count = __shfl_sync(0xffffffffu, total_count_reduced, 0);
    if (lane == 0) {
        local_neighbor_counts[q] = total_count;
    }

    if (total_count == 0) {
        warp_global_softmax_fallback(
            lane,
            q,
            anchor_count,
            inv_two_sigma2,
            qx,
            qy,
            qz,
            anchors,
            anchor_flow,
            output_flow);
        return;
    }

    merge_and_write_warp_softmax(
        lane,
        q,
        local_max,
        local_denominator,
        local_flow_x,
        local_flow_y,
        local_flow_z,
        output_flow);
}

}  // namespace

void gaussian_softmax_recovery_kernel_launcher(
    int query_count,
    int anchor_count,
    float sigma,
    const float *queries,
    const float *anchors,
    const float *anchor_flow,
    float *output_flow,
    cudaStream_t stream) {
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);
    gaussian_softmax_recovery_warp_kernel<<<blocks, RECOVERY_THREADS, 0, stream>>>(
        query_count,
        anchor_count,
        inv_two_sigma2,
        queries,
        anchors,
        anchor_flow,
        output_flow);
}

void gaussian_recovery_hash_build_kernel_launcher(
    int anchor_count,
    int hash_size,
    float cell_size,
    const float *anchors,
    int *hash_heads,
    int *anchor_next,
    int *anchor_cells,
    cudaStream_t stream) {
    constexpr int THREADS = 256;
    const int blocks = (anchor_count + THREADS - 1) / THREADS;
    const float inv_cell_size = 1.0f / cell_size;
    gaussian_recovery_hash_build_kernel<<<blocks, THREADS, 0, stream>>>(
        anchor_count,
        hash_size,
        inv_cell_size,
        anchors,
        hash_heads,
        anchor_next,
        anchor_cells);
}

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
    cudaStream_t stream) {
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);
    const float radius2 = radius * radius;
    const float inv_cell_size = 1.0f / cell_size;
    gaussian_softmax_recovery_local_kernel<<<blocks, RECOVERY_THREADS, 0, stream>>>(
        query_count,
        anchor_count,
        hash_size,
        inv_two_sigma2,
        radius2,
        inv_cell_size,
        queries,
        anchors,
        anchor_flow,
        hash_heads,
        anchor_next,
        anchor_cells,
        output_flow,
        local_neighbor_counts);
}
