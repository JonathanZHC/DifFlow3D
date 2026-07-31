"""Isaac Sim 6 multi-camera RGB-D + fused point-cloud smoke test.

The script uses the Isaac Sim 6 experimental RTX camera API:
  * multiple RtxCamera prims
  * TiledCameraSensor for batched RGB/depth/instance rendering
  * CUDA-resident Warp outputs converted to PyTorch
  * configurable RGB/depth corruption
  * GPU depth unprojection and world-frame multi-camera fusion

Run inside the image with:
    /isaac-sim/python.sh /workspace/isaacsim_multicamera_rgbd.py \
        --headless --cameras 4 --frames 60 --corrupt

This is a sensor/frontend validation script. Replace the simple USD scene with
an imported FR3/scene USD and pass `fused_points_world` directly to the existing
voxel/FPS/DifFlow3D pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# SimulationApp must be created before importing most Isaac/Omniverse modules.
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-hz", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--corrupt", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/camera_output"))
    parser.add_argument("--save-last-frame", action="store_true")
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--depth-noise-base-m", type=float, default=0.0015)
    parser.add_argument("--depth-noise-quadratic", type=float, default=0.0010)
    parser.add_argument("--depth-dropout", type=float, default=0.01)
    parser.add_argument("--edge-dropout", type=float, default=0.20)
    parser.add_argument("--depth-quantization-m", type=float, default=0.001)
    parser.add_argument("--rgb-noise-std", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


ARGS = parse_args()
if ARGS.cameras < 1:
    raise ValueError("--cameras must be positive")
if ARGS.width < 16 or ARGS.height < 16:
    raise ValueError("Camera resolution is too small")

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": ARGS.width,
        "height": ARGS.height,
        "renderer": "RaytracedLighting",
    }
)

import torch
import warp as wp
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.sensors.experimental.rtx import RtxCamera, TiledCameraSensor


@dataclass(frozen=True)
class CameraCalibration:
    position_world: torch.Tensor  # [3]
    rotation_world_camera_usd: torch.Tensor  # [3,3]
    fx: float
    fy: float
    cx: float
    cy: float


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def look_at_rotation_world_camera(
    eye: np.ndarray,
    target: np.ndarray,
    up_world: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Return local-camera-to-world rotation for a USD camera.

    USD camera convention: +X right, +Y up, -Z optical forward.
    """
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up_world))
    up = normalize(np.cross(right, forward))
    return np.stack((right, up, -forward), axis=1)


def rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    # Stable matrix-to-quaternion conversion without a SciPy dependency here.
    m = matrix
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        diagonal = np.diag(m)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def add_cube(
    stage,
    path: str,
    translation: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*translation))
    xform.SetScale(Gf.Vec3f(*scale))


def build_scene() -> None:
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    add_cube(stage, "/World/Ground", (0.0, 0.0, -0.05), (4.0, 4.0, 0.10))
    add_cube(stage, "/World/BoxA", (0.0, 0.0, 0.35), (0.45, 0.35, 0.70))
    add_cube(stage, "/World/BoxB", (0.75, 0.25, 0.25), (0.30, 0.55, 0.50))
    add_cube(stage, "/World/BoxC", (-0.65, -0.30, 0.45), (0.40, 0.25, 0.90))

    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(900.0)
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(1.0)


def create_cameras() -> tuple[TiledCameraSensor, list[CameraCalibration]]:
    radius = 2.0
    height = 1.25
    target = np.array([0.0, 0.0, 0.35], dtype=np.float64)

    focal_length_mm = 18.0
    horizontal_aperture_mm = 24.0
    vertical_aperture_mm = horizontal_aperture_mm * ARGS.height / ARGS.width
    fx = focal_length_mm / horizontal_aperture_mm * ARGS.width
    fy = focal_length_mm / vertical_aperture_mm * ARGS.height
    cx = (ARGS.width - 1.0) * 0.5
    cy = (ARGS.height - 1.0) * 0.5

    paths: list[str] = []
    calibrations: list[CameraCalibration] = []

    for camera_index in range(ARGS.cameras):
        angle = 2.0 * np.pi * camera_index / ARGS.cameras
        eye = np.array(
            [radius * np.cos(angle), radius * np.sin(angle), height],
            dtype=np.float64,
        )
        rotation = look_at_rotation_world_camera(eye, target)
        quaternion = rotation_matrix_to_quaternion_wxyz(rotation)
        path = f"/World/Camera_{camera_index}"

        camera = RtxCamera(
            path,
            tick_rate=ARGS.camera_hz,
            positions=eye[None, :],
            orientations=quaternion[None, :],
        )
        camera.camera.set_focal_lengths(focal_length_mm)
        camera.camera.set_horizontal_apertures(horizontal_aperture_mm)
        camera.camera.set_vertical_apertures(vertical_aperture_mm)
        camera.camera.set_clipping_ranges(0.05, ARGS.max_depth_m)

        paths.append(path)
        calibrations.append(
            CameraCalibration(
                position_world=torch.tensor(eye, dtype=torch.float32, device="cuda"),
                rotation_world_camera_usd=torch.tensor(
                    rotation, dtype=torch.float32, device="cuda"
                ),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )
        )

    sensor = TiledCameraSensor(
        paths,
        resolution=(ARGS.height, ARGS.width),
        annotators=[
            "rgb",
            "distance_to_image_plane",
            "instance_id_segmentation",
        ],
    )
    return sensor, calibrations


def warp_to_torch(array: wp.array | None) -> torch.Tensor:
    if array is None:
        raise RuntimeError("Camera annotator has not produced data yet")
    tensor = wp.to_torch(array)
    if tensor.device.type != "cuda":
        tensor = tensor.to("cuda", non_blocking=True)
    return tensor


def corrupt_rgb(rgb_u8: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    rgb = rgb_u8[..., :3].float() / 255.0
    camera_count = rgb.shape[0]
    brightness = torch.empty(
        (camera_count, 1, 1, 1), device=rgb.device
    ).uniform_(0.80, 1.20, generator=generator)
    gamma = torch.empty(
        (camera_count, 1, 1, 1), device=rgb.device
    ).uniform_(0.85, 1.20, generator=generator)
    rgb = torch.clamp(rgb * brightness, 0.0, 1.0)
    rgb = torch.pow(rgb.clamp_min(1.0e-6), gamma)
    noise = torch.randn(
        rgb.shape, device=rgb.device, dtype=rgb.dtype, generator=generator
    ) * ARGS.rgb_noise_std
    return torch.clamp(rgb + noise, 0.0, 1.0)


def corrupt_depth(depth: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    valid = torch.isfinite(depth) & (depth > 0.0) & (depth < ARGS.max_depth_m)
    clean = torch.where(valid, depth, torch.zeros_like(depth))

    sigma = ARGS.depth_noise_base_m + ARGS.depth_noise_quadratic * clean.square()
    noisy = clean + torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    ) * sigma

    if ARGS.depth_quantization_m > 0.0:
        noisy = (
            torch.round(noisy / ARGS.depth_quantization_m)
            * ARGS.depth_quantization_m
        )

    # Depth discontinuities receive additional dropout, producing realistic
    # missing/flying-edge regions without changing the clean GT buffer.
    dx = torch.zeros_like(noisy)
    dy = torch.zeros_like(noisy)
    dx[..., :, 1:] = (clean[..., :, 1:] - clean[..., :, :-1]).abs()
    dy[..., 1:, :] = (clean[..., 1:, :] - clean[..., :-1, :]).abs()
    edge = torch.maximum(dx, dy) > 0.025

    uniform = torch.rand(
        noisy.shape, device=noisy.device, dtype=noisy.dtype, generator=generator
    )
    dropout_probability = ARGS.depth_dropout + edge.float() * ARGS.edge_dropout
    keep = uniform >= dropout_probability
    return torch.where(valid & keep, noisy.clamp_min(0.0), torch.zeros_like(noisy))


def depth_to_world_pointcloud(
    depth_batch: torch.Tensor,
    rgb_batch: torch.Tensor,
    calibrations: list[CameraCalibration],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unproject distance-to-image-plane into a fused world-frame PCD."""
    device = depth_batch.device
    v, u = torch.meshgrid(
        torch.arange(ARGS.height, device=device, dtype=torch.float32),
        torch.arange(ARGS.width, device=device, dtype=torch.float32),
        indexing="ij",
    )

    world_points: list[torch.Tensor] = []
    world_colors: list[torch.Tensor] = []
    for camera_index, calibration in enumerate(calibrations):
        z = depth_batch[camera_index]
        valid = torch.isfinite(z) & (z > 0.0) & (z < ARGS.max_depth_m)
        x_cv = (u - calibration.cx) / calibration.fx * z
        y_cv = (v - calibration.cy) / calibration.fy * z

        # OpenCV camera frame (+x right, +y down, +z forward) -> USD camera
        # frame (+x right, +y up, -z forward).
        points_usd = torch.stack((x_cv, -y_cv, -z), dim=-1)
        points_world = (
            points_usd @ calibration.rotation_world_camera_usd.transpose(0, 1)
            + calibration.position_world
        )
        world_points.append(points_world[valid])
        world_colors.append(rgb_batch[camera_index][valid])

    return torch.cat(world_points, dim=0), torch.cat(world_colors, dim=0)


def save_last_frame(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    points: torch.Tensor,
    colors: torch.Tensor,
    instance_ids: torch.Tensor,
) -> None:
    from PIL import Image

    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_cpu = (rgb[0].clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    Image.fromarray(rgb_cpu).save(ARGS.output_dir / "camera_0_rgb.png")

    depth_cpu = depth[0].cpu().numpy()
    np.save(ARGS.output_dir / "camera_0_depth.npy", depth_cpu)
    np.savez_compressed(
        ARGS.output_dir / "fused_pointcloud.npz",
        points=points.cpu().numpy().astype(np.float32),
        colors=colors.cpu().numpy().astype(np.float32),
    )
    np.save(
        ARGS.output_dir / "instance_ids.npy",
        instance_ids.cpu().numpy(),
    )


def main() -> None:
    torch.manual_seed(ARGS.seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(ARGS.seed)

    build_scene()
    sensor, calibrations = create_cameras()
    app_utils.play(commit=True)

    for _ in range(ARGS.warmup_frames):
        simulation_app.update()

    last_payload = None
    for frame_index in range(ARGS.frames):
        simulation_app.update()

        rgb_wp, _ = sensor.get_data("rgb")
        depth_wp, _ = sensor.get_data("distance_to_image_plane")
        instance_wp, _ = sensor.get_data("instance_id_segmentation")

        rgb_raw = warp_to_torch(rgb_wp)
        depth_clean = warp_to_torch(depth_wp).float()
        if depth_clean.ndim == 4 and depth_clean.shape[-1] == 1:
            depth_clean = depth_clean.squeeze(-1)
        if depth_clean.ndim != 3:
            raise RuntimeError(
                f"Expected tiled depth [C,H,W], got {tuple(depth_clean.shape)}"
            )

        instance_ids = warp_to_torch(instance_wp)
        if instance_ids.ndim == 4 and instance_ids.shape[-1] == 1:
            instance_ids = instance_ids.squeeze(-1)

        if ARGS.corrupt:
            rgb = corrupt_rgb(rgb_raw, generator)
            depth = corrupt_depth(depth_clean, generator)
        else:
            rgb = rgb_raw[..., :3].float() / 255.0
            depth = depth_clean

        fused_points_world, fused_colors = depth_to_world_pointcloud(
            depth,
            rgb,
            calibrations,
        )
        last_payload = (
            rgb,
            depth,
            fused_points_world,
            fused_colors,
            instance_ids,
        )

        if frame_index == 0 or (frame_index + 1) % 10 == 0:
            print(
                f"frame={frame_index + 1:04d} "
                f"rgb={tuple(rgb.shape)} "
                f"depth={tuple(depth.shape)} "
                f"fused_points={fused_points_world.shape[0]} "
                f"device={fused_points_world.device}"
            )

    if ARGS.save_last_frame and last_payload is not None:
        save_last_frame(*last_payload)
        print(f"Saved final observation to {ARGS.output_dir}")


try:
    main()
finally:
    simulation_app.close()
