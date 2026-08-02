#!/usr/bin/env python3
"""Run the two-camera Isaac Sim RGB-D and point-cloud publisher."""

from __future__ import annotations

import argparse
import os
import time
import traceback


WARMUP_FRAMES = 20
ROS_DOMAIN_ID = 117


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish two independent RGB-D camera streams and "
            "full per-camera point clouds forever."
        )
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--rgbd-hz",
        "--camera-hz",
        dest="rgbd_hz",
        type=float,
        default=30.0,
        help=(
            "RGB, 32FC1 depth, CameraInfo and pose publication rate. "
            "The legacy --camera-hz name is kept as an alias."
        ),
    )
    parser.add_argument(
        "--pointcloud-hz",
        type=float,
        default=2.0,
        help=(
            "Full per-camera PointCloud2 publication rate."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim without a GUI viewport.",
    )
    parser.add_argument(
        "--corrupt",
        action="store_true",
        help="Enable NVIDIA Warp GPU camera corruption.",
    )
    parser.add_argument(
        "--no-rgb-corruption",
        dest="rgb_corruption",
        action="store_false",
        default=True,
        help=(
            "Keep RGB clean when --corrupt is enabled. "
            "RGB corruption is enabled by default."
        ),
    )
    parser.add_argument(
        "--no-depth-corruption",
        dest="depth_corruption",
        action="store_false",
        default=True,
        help=(
            "Keep depth and point clouds clean when --corrupt is enabled. "
            "Depth corruption is enabled by default."
        ),
    )
    parser.add_argument(
        "--scene",
        "--scene-mode",
        dest="scene_mode",
        choices=("static", "dynamic"),
        default="static",
        help=(
            "Select the complete scene configuration: "
            "'static' restores the original stationary tabletop; "
            "'dynamic' uses the tall shelf, bottle and floating object. "
            "Default: static."
        ),
    )
    parser.add_argument(
        "--motion-speed-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply all moving-object trajectory speeds. "
            "Default: 1.0."
        ),
    )
    parser.add_argument(
        "--profile-every",
        type=int,
        default=60,
        help="Print timing statistics every N RGB-D frames.",
    )
    return parser.parse_args()


ARGS = parse_args()

if ARGS.width <= 0 or ARGS.height <= 0:
    raise ValueError("Width and height must be positive.")
if ARGS.rgbd_hz <= 0.0:
    raise ValueError("rgbd-hz must be positive.")
if ARGS.pointcloud_hz <= 0.0:
    raise ValueError("pointcloud-hz must be positive.")
if ARGS.pointcloud_hz > ARGS.rgbd_hz:
    raise ValueError(
        "pointcloud-hz cannot exceed rgbd-hz."
    )
if ARGS.motion_speed_scale <= 0.0:
    raise ValueError("motion-speed-scale must be positive.")
if ARGS.profile_every <= 0:
    raise ValueError("profile-every must be positive.")
if (
    ARGS.corrupt
    and not ARGS.rgb_corruption
    and not ARGS.depth_corruption
):
    raise ValueError(
        "--corrupt cannot be combined with both "
        "--no-rgb-corruption and --no-depth-corruption."
    )

os.environ["ROS_DOMAIN_ID"] = str(ROS_DOMAIN_ID)
os.environ.setdefault(
    "RMW_IMPLEMENTATION",
    "rmw_fastrtps_cpp",
)

# SimulationApp must be created before importing omni, pxr or Replicator.
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": ARGS.width,
        "height": ARGS.height,
    }
)

import numpy as np
import omni.timeline
import omni.usd

from camera_pyramid_visualizer import CameraPyramidVisualizer
from camera_settings import (
    CameraRigConfig,
    CorruptionConfig,
    capture_all_cameras,
    create_cameras,
)
from moving_objects import MovingObjectController
from ros_camera_publisher import RosCameraPublisher
from scene_settings import (
    DEFAULT_SCENE_CONFIG,
    build_scene,
)


def _print_profile(
    samples: list[dict[str, float]],
) -> None:
    print("[PROFILE]", flush=True)
    for key in (
        "capture_ms",
        "ros_ms",
        "pointcloud_ms",
        "pipeline_ms",
        "actual_period_ms",
    ):
        values = np.asarray(
            [sample[key] for sample in samples],
            dtype=np.float64,
        )
        if key == "actual_period_ms":
            values = values[values > 0.0]
        if values.size == 0:
            continue

        print(
            f"  {key:<18}"
            f" mean={values.mean():8.3f}"
            f" p95={np.percentile(values, 95):8.3f}"
            f" max={values.max():8.3f}",
            flush=True,
        )

    period_values = np.asarray(
        [
            sample["actual_period_ms"]
            for sample in samples
            if sample["actual_period_ms"] > 0.0
        ],
        dtype=np.float64,
    )
    if period_values.size:
        print(
            f"  achieved_hz       "
            f"{1000.0 / period_values.mean():8.3f}",
            flush=True,
        )


def main() -> None:
    ros_publisher = None
    pyramid_visualizer = None
    moving_object_controller = None

    try:
        stage = omni.usd.get_context().get_stage()
        scene = build_scene(
            stage,
            scene_mode=ARGS.scene_mode,
        )

        if ARGS.scene_mode == "dynamic":
            moving_object_controller = MovingObjectController(
                stage,
                table_surface_z=scene.table_surface_z,
                speed_scale=ARGS.motion_speed_scale,
            )

        rig = CameraRigConfig(
            width=ARGS.width,
            height=ARGS.height,
        )
        corruption = CorruptionConfig(
            enabled=ARGS.corrupt,
            corrupt_rgb=ARGS.rgb_corruption,
            corrupt_depth=ARGS.depth_corruption,
        )
        corruption.validate()

        cameras = create_cameras(
            stage,
            rig,
            corruption,
        )

        if not ARGS.headless:
            pyramid_visualizer = CameraPyramidVisualizer(
                cameras,
                ARGS.width,
                ARGS.height,
            )

        ros_publisher = RosCameraPublisher(
            cameras,
            rgbd_hz=ARGS.rgbd_hz,
            pointcloud_hz=ARGS.pointcloud_hz,
            max_depth_m=rig.max_depth_m,
            world_frame_id=rig.world_frame_id,
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        print(
            f"Warming up the renderer for {WARMUP_FRAMES} frames.",
            flush=True,
        )
        for _ in range(WARMUP_FRAMES):
            simulation_app.update()

        stream = "corrupted" if ARGS.corrupt else "clean"
        print(
            "Running forever: "
            f"resolution={ARGS.width}x{ARGS.height}, "
            f"rgbd_hz={ARGS.rgbd_hz}, "
            f"pointcloud_hz={ARGS.pointcloud_hz}, "
            f"headless={ARGS.headless}, "
            f"scene={ARGS.scene_mode}, "
            f"motion_speed_scale={ARGS.motion_speed_scale}, "
            f"stream={stream}, "
            f"rgb_corruption="
            f"{ARGS.corrupt and ARGS.rgb_corruption}, "
            f"depth_corruption="
            f"{ARGS.corrupt and ARGS.depth_corruption}, "
            "depth_encoding=32FC1, "
            "pointcloud=separate_full_optical_frame, "
            "downsampling=false, fusion=false",
            flush=True,
        )

        publish_period = 1.0 / ARGS.rgbd_hz
        motion_start_time = time.perf_counter()
        next_publish_time = motion_start_time
        published_frames = 0
        last_publish_time = None
        profile_samples: list[dict[str, float]] = []
        log_every_frames = max(1, int(round(ARGS.rgbd_hz)))

        while simulation_app.is_running():
            if moving_object_controller is not None:
                motion_now = time.perf_counter()
                moving_object_controller.update(
                    motion_now - motion_start_time
                )

            # Render after updating USD transforms so the captured RGB-D frame
            # contains the current moving-object positions.
            simulation_app.update()

            now = time.perf_counter()
            if now < next_publish_time:
                continue

            next_publish_time = max(
                next_publish_time + publish_period,
                now,
            )

            pipeline_start = time.perf_counter()
            frames = capture_all_cameras(
                cameras,
                rig,
                corruption,
                frame_index=published_frames,
            )
            capture_end = time.perf_counter()

            pointcloud_ms = ros_publisher.publish(frames)
            ros_end = time.perf_counter()

            if last_publish_time is None:
                actual_period_ms = 0.0
            else:
                actual_period_ms = 1000.0 * (
                    ros_end - last_publish_time
                )
            last_publish_time = ros_end

            profile_samples.append(
                {
                    "capture_ms": 1000.0 * (
                        capture_end - pipeline_start
                    ),
                    "ros_ms": 1000.0 * (
                        ros_end - capture_end
                    ),
                    "pointcloud_ms": pointcloud_ms,
                    "pipeline_ms": 1000.0 * (
                        ros_end - pipeline_start
                    ),
                    "actual_period_ms": actual_period_ms,
                }
            )

            published_frames += 1

            if published_frames % log_every_frames == 0:
                point_counts = ", ".join(
                    f"{name}={count}"
                    for name, count
                    in ros_publisher.last_point_counts.items()
                )
                print(
                    f"frame={published_frames:06d} "
                    f"points[{point_counts}] "
                    f"pointcloud_ms="
                    f"{ros_publisher.last_pointcloud_ms:.3f}",
                    flush=True,
                )

            if len(profile_samples) >= ARGS.profile_every:
                _print_profile(profile_samples)
                profile_samples.clear()

    except KeyboardInterrupt:
        print("Interrupted by the user.", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if ros_publisher is not None:
            ros_publisher.shutdown()
        if pyramid_visualizer is not None:
            pyramid_visualizer.clear()

        try:
            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass

        simulation_app.close()


if __name__ == "__main__":
    main()
