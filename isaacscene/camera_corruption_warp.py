#!/usr/bin/env python3
"""NVIDIA Warp kernels used by the camera pipeline."""

from __future__ import annotations

import warp as wp


@wp.kernel
def rgb_camera_corruption_wp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    noise_std_255: float,
    exposure_fraction: float,
    seed: int,
):
    """Apply exposure variation and independent RGB Gaussian noise."""

    row, col = wp.tid()
    height = data_in.shape[0]
    width = data_in.shape[1]
    pixel_count = height * width
    pixel_id = row * width + col

    exposure_state = wp.rand_init(seed, pixel_count * 4)
    exposure = 1.0 + exposure_fraction * (
        2.0 * wp.randf(exposure_state) - 1.0
    )

    state_r = wp.rand_init(seed, pixel_id + pixel_count * 0)
    state_g = wp.rand_init(seed, pixel_id + pixel_count * 1)
    state_b = wp.rand_init(seed, pixel_id + pixel_count * 2)

    red = (
        wp.float32(data_in[row, col, 0]) * exposure
        + noise_std_255 * wp.randn(state_r)
    )
    green = (
        wp.float32(data_in[row, col, 1]) * exposure
        + noise_std_255 * wp.randn(state_g)
    )
    blue = (
        wp.float32(data_in[row, col, 2]) * exposure
        + noise_std_255 * wp.randn(state_b)
    )

    data_out[row, col, 0] = wp.uint8(
        wp.clamp(red, 0.0, 255.0)
    )
    data_out[row, col, 1] = wp.uint8(
        wp.clamp(green, 0.0, 255.0)
    )
    data_out[row, col, 2] = wp.uint8(
        wp.clamp(blue, 0.0, 255.0)
    )

    if data_out.shape[2] > 3:
        data_out[row, col, 3] = data_in[row, col, 3]


@wp.func
def _is_depth_edge(
    data_in: wp.array2d(dtype=wp.float32),
    row: int,
    col: int,
    center: float,
    threshold: float,
) -> bool:
    """Return whether a valid four-neighbour depth jump is large."""

    height = data_in.shape[0]
    width = data_in.shape[1]
    edge = False

    if col > 0:
        value = data_in[row, col - 1]
        if wp.isfinite(value) and value > 0.0:
            edge = edge or wp.abs(value - center) > threshold

    if col + 1 < width:
        value = data_in[row, col + 1]
        if wp.isfinite(value) and value > 0.0:
            edge = edge or wp.abs(value - center) > threshold

    if row > 0:
        value = data_in[row - 1, col]
        if wp.isfinite(value) and value > 0.0:
            edge = edge or wp.abs(value - center) > threshold

    if row + 1 < height:
        value = data_in[row + 1, col]
        if wp.isfinite(value) and value > 0.0:
            edge = edge or wp.abs(value - center) > threshold

    return edge


@wp.kernel
def depth_camera_corruption_wp(
    data_in: wp.array2d(dtype=wp.float32),
    data_out: wp.array2d(dtype=wp.float32),
    noise_base_m: float,
    noise_quadratic: float,
    quantization_m: float,
    random_dropout_probability: float,
    edge_dropout_probability: float,
    edge_threshold_m: float,
    seed: int,
):
    """Apply depth noise, quantization, dropout and edge dropout."""

    row, col = wp.tid()
    width = data_in.shape[1]
    pixel_id = row * width + col
    clean_depth = data_in[row, col]

    # distance_to_image_plane uses zero for infinity/background.
    valid = wp.isfinite(clean_depth) and clean_depth > 0.0
    if not valid:
        data_out[row, col] = wp.float32(wp.NAN)
        return

    state = wp.rand_init(seed, pixel_id)
    sigma = (
        noise_base_m
        + noise_quadratic * clean_depth * clean_depth
    )
    noisy_depth = clean_depth + sigma * wp.randn(state)

    if quantization_m > 0.0:
        noisy_depth = (
            wp.round(noisy_depth / quantization_m)
            * quantization_m
        )

    random_dropout = (
        wp.randf(state) < random_dropout_probability
    )
    edge_dropout = (
        _is_depth_edge(
            data_in,
            row,
            col,
            clean_depth,
            edge_threshold_m,
        )
        and wp.randf(state) < edge_dropout_probability
    )

    if (
        wp.isfinite(noisy_depth)
        and noisy_depth > 0.0
        and not random_dropout
        and not edge_dropout
    ):
        data_out[row, col] = noisy_depth
    else:
        data_out[row, col] = wp.float32(wp.NAN)
