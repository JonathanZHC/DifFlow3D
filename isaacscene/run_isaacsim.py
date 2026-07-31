#!/usr/bin/env python3
"""启动 Isaac Sim，组合场景、相机和 ROS 发布器。"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-hz", type=float, default=30.0)
    parser.add_argument("--warmup-frames", type=int, default=40)

    # 0 表示一直运行到关闭 GUI 或 Ctrl+C。
    parser.add_argument("--frames", type=int, default=0)

    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=5.0)

    parser.add_argument("--corrupt", action="store_true")
    parser.add_argument(
        "--publish-clean",
        action="store_true",
        help="在 corrupt 流之外同时发布 clean 话题",
    )
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--no-ros", action="store_true")
    parser.add_argument("--ros-domain-id", type=int, default=100)

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="不发布 RViz 相机视锥和纹理图像平面",
    )
    parser.add_argument(
        "--visualization-hz",
        type=float,
        default=2.0,
        help="三维相机纹理平面更新频率；点云仍按 camera-hz 发布",
    )
    parser.add_argument(
        "--frustum-depth",
        type=float,
        default=0.45,
        help="RViz 相机视锥图像平面距离，单位米",
    )
    parser.add_argument(
        "--texture-jpeg-quality",
        type=int,
        default=75,
        help="RViz 三维图像纹理 JPEG 质量 [1,100]",
    )

    parser.add_argument(
        "--no-isaac-visualization",
        action="store_true",
        help="关闭 Isaac Sim 主视口点云、相机视锥和图像窗口",
    )
    parser.add_argument(
        "--isaac-visualization-hz",
        type=float,
        default=5.0,
        help="Isaac Sim 内点云和图像窗口更新频率",
    )
    parser.add_argument(
        "--isaac-max-points",
        type=int,
        default=40000,
        help="Isaac Sim 主视口最多显示的融合点数",
    )
    parser.add_argument(
        "--isaac-point-size",
        type=float,
        default=0.008,
        help="Isaac Sim 点云显示点直径，单位米",
    )
    parser.add_argument(
        "--isaac-frustum-depth",
        type=float,
        default=0.45,
        help="Isaac Sim 相机视锥长度，单位米",
    )
    parser.add_argument(
        "--no-isaac-camera-windows",
        action="store_true",
        help="保留三维点云/视锥，但不创建相机 RGB 窗口",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/camera_output/isaacscene"),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="每 N 个发布帧保存一次，0 表示不保存帧",
    )

    return parser.parse_args()


ARGS = parse_args()

if ARGS.width <= 0 or ARGS.height <= 0:
    raise ValueError("width 和 height 必须大于 0")
if ARGS.camera_hz <= 0.0:
    raise ValueError("camera-hz 必须大于 0")
if ARGS.frames < 0:
    raise ValueError("frames 不得为负数")
if ARGS.point_stride <= 0:
    raise ValueError("point-stride 必须大于 0")
if ARGS.max_depth <= 0.0:
    raise ValueError("max-depth 必须大于 0")
if ARGS.save_every < 0:
    raise ValueError("save-every 不得为负数")
if ARGS.visualization_hz <= 0.0:
    raise ValueError("visualization-hz 必须大于 0")
if ARGS.frustum_depth <= 0.0:
    raise ValueError("frustum-depth 必须大于 0")
if not 1 <= ARGS.texture_jpeg_quality <= 100:
    raise ValueError("texture-jpeg-quality 必须在 [1,100]")
if ARGS.isaac_visualization_hz <= 0.0:
    raise ValueError("isaac-visualization-hz 必须大于 0")
if ARGS.isaac_max_points <= 0:
    raise ValueError("isaac-max-points 必须大于 0")
if ARGS.isaac_point_size <= 0.0:
    raise ValueError("isaac-point-size 必须大于 0")
if ARGS.isaac_frustum_depth <= 0.0:
    raise ValueError("isaac-frustum-depth 必须大于 0")

os.environ["ROS_DOMAIN_ID"] = str(ARGS.ros_domain_id)
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

# 必须先创建 SimulationApp，再导入 omni/pxr/Replicator 模块。
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": ARGS.renderer,
        "width": ARGS.width,
        "height": ARGS.height,
    }
)


import numpy as np
import omni.timeline
import omni.usd

from camera_settings import (
    CameraRigConfig,
    CorruptionConfig,
    capture_all_cameras,
    create_cameras,
    fuse_world_pointcloud,
    save_calibration,
    save_frame_bundle,
)
from scene_settings import build_scene


def main() -> None:
    ros_publisher = None
    isaac_visualizer = None

    stage = omni.usd.get_context().get_stage()
    object_paths = build_scene(stage)

    rig = CameraRigConfig(
        width=ARGS.width,
        height=ARGS.height,
        max_depth_m=ARGS.max_depth,
        point_stride=ARGS.point_stride,
    )
    corruption = CorruptionConfig(
        enabled=ARGS.corrupt,
        seed=ARGS.seed,
    )
    rng = np.random.default_rng(corruption.seed)

    cameras = create_cameras(stage, rig)

    if (
        not ARGS.headless
        and not ARGS.no_isaac_visualization
    ):
        from isaacsim_visualizer import IsaacSimVisualizer

        isaac_visualizer = IsaacSimVisualizer(
            stage=stage,
            cameras=cameras,
            image_width=rig.width,
            image_height=rig.height,
            primary_stream_label=(
                "corrupted" if ARGS.corrupt else "clean"
            ),
            update_hz=ARGS.isaac_visualization_hz,
            max_points=ARGS.isaac_max_points,
            point_size_m=ARGS.isaac_point_size,
            frustum_depth_m=ARGS.isaac_frustum_depth,
            show_image_windows=(
                not ARGS.no_isaac_camera_windows
            ),
        )

    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    save_calibration(ARGS.output_dir, cameras, rig)
    stage.GetRootLayer().Export(
        str(ARGS.output_dir / "tabletop_scene.usda")
    )

    print("\n场景对象：", flush=True)
    for name, path in object_paths.items():
        print(f"  {name}: {path}", flush=True)

    print("\n相机标定：", flush=True)
    for camera in cameras:
        print(
            f"\n{camera.spec.name}"
            f"\n  prim={camera.spec.prim_path}"
            f"\n  optical_frame={camera.spec.optical_frame_id}"
            f"\n  K=\n{camera.K}"
            f"\n  T_world_from_camera_optical="
            f"\n{camera.T_world_from_camera_optical}",
            flush=True,
        )

    if not ARGS.no_ros:
        from ros_camera_publisher import RosCameraPublisher

        ros_publisher = RosCameraPublisher(
            cameras=cameras,
            world_frame_id=rig.world_frame_id,
            publish_clean=ARGS.publish_clean,
            enable_visualization=not ARGS.no_visualization,
            visualization_hz=ARGS.visualization_hz,
            frustum_depth_m=ARGS.frustum_depth,
            texture_jpeg_quality=ARGS.texture_jpeg_quality,
            primary_stream_label=(
                "corrupted" if ARGS.corrupt else "clean"
            ),
        )

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    print(
        f"\n预热渲染器：{ARGS.warmup_frames} 帧",
        flush=True,
    )
    for _ in range(ARGS.warmup_frames):
        simulation_app.update()

    publish_period = 1.0 / ARGS.camera_hz
    next_publish_time = time.perf_counter()
    published_frames = 0

    print(
        "\n开始运行："
        f" headless={ARGS.headless},"
        f" camera_hz={ARGS.camera_hz},"
        f" corruption={ARGS.corrupt},"
        f" ROS={not ARGS.no_ros},"
        f" rviz_visualization={not ARGS.no_visualization},"
        f" isaac_visualization="
        f"{not ARGS.no_isaac_visualization}",
        flush=True,
    )

    while simulation_app.is_running():
        simulation_app.update()

        now = time.perf_counter()
        if now < next_publish_time:
            # GUI 仍需持续 update，不在这里长时间 sleep。
            continue

        # 避免执行速度低于目标频率后持续追赶。
        next_publish_time = max(
            next_publish_time + publish_period,
            now,
        )

        frames = capture_all_cameras(
            cameras,
            rig,
            corruption,
            rng,
        )
        fused_points, fused_colors = fuse_world_pointcloud(
            frames,
            clean=False,
        )

        clean_fused_points = None
        clean_fused_colors = None
        if ARGS.publish_clean:
            clean_fused_points, clean_fused_colors = (
                fuse_world_pointcloud(frames, clean=True)
            )

        if ros_publisher is not None:
            ros_publisher.publish(
                frames,
                fused_points,
                fused_colors,
                clean_fused_points,
                clean_fused_colors,
            )

        if isaac_visualizer is not None:
            isaac_visualizer.update(
                frames,
                fused_points,
                fused_colors,
            )

        if (
            ARGS.save_every > 0
            and published_frames % ARGS.save_every == 0
        ):
            saved_dir = save_frame_bundle(
                ARGS.output_dir,
                published_frames,
                frames,
            )
            print(f"保存：{saved_dir}", flush=True)

        if published_frames % 30 == 0:
            camera_stats = ", ".join(
                f"{name}={frame.points_world.shape[0]}"
                for name, frame in frames.items()
            )
            print(
                f"frame={published_frames:06d} "
                f"points[{camera_stats}] "
                f"fused={fused_points.shape[0]}",
                flush=True,
            )

        published_frames += 1
        if ARGS.frames > 0 and published_frames >= ARGS.frames:
            break

    if isaac_visualizer is not None:
        isaac_visualizer.shutdown()

    if ros_publisher is not None:
        ros_publisher.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在关闭。", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
