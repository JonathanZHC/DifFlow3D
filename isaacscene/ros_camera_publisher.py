#!/usr/bin/env python3
"""发布 RGB-D、点云、相机标定、相机位姿、TF 和 RViz 可视化。

主数据流的语义：
  - 启动参数包含 --corrupt 时，主话题发布损坏后的 RGB-D/点云；
  - 未包含 --corrupt 时，主话题发布干净 RGB-D/点云。

固定主话题：
  /camera_N/color/image_raw
  /camera_N/color/camera_info
  /camera_N/depth/image_raw
  /camera_N/points
  /camera_N/pose
  /cameras/fused_points

可选干净数据：
  /camera_N/clean/color/image_raw
  /camera_N/clean/color/camera_info
  /camera_N/clean/depth/image_raw
  /camera_N/clean/points
  /cameras/clean/fused_points

三维可视化：
  /cameras/visualization  visualization_msgs/MarkerArray

静态 TF：
  world -> camera_N_optical_frame
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import Point, PoseStamped, TransformStamped
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import (
        CameraInfo,
        Image,
        PointCloud2,
        PointField,
    )
    from std_msgs.msg import ColorRGBA
    from tf2_ros import StaticTransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError as error:
    raise RuntimeError(
        "无法导入 ROS 2 可视化模块。请确认容器中安装了："
        " ros-jazzy-rclpy ros-jazzy-sensor-msgs"
        " ros-jazzy-geometry-msgs ros-jazzy-visualization-msgs"
        " ros-jazzy-tf2-ros。"
    ) from error

from camera_settings import (
    CameraFrame,
    CameraRuntime,
    rotation_matrix_to_quaternion_xyzw,
)


@dataclass
class _CameraPublishers:
    color: Any
    depth: Any
    camera_info: Any
    color_camera_info: Any
    points: Any
    pose: Any
    clean_color: Any | None = None
    clean_depth: Any | None = None
    clean_camera_info: Any | None = None
    clean_points: Any | None = None


def _point(x: float, y: float, z: float) -> Point:
    message = Point()
    message.x = float(x)
    message.y = float(y)
    message.z = float(z)
    return message


def _color(
    red: float,
    green: float,
    blue: float,
    alpha: float = 1.0,
) -> ColorRGBA:
    message = ColorRGBA()
    message.r = float(red)
    message.g = float(green)
    message.b = float(blue)
    message.a = float(alpha)
    return message


class RosCameraPublisher:
    def __init__(
        self,
        cameras: list[CameraRuntime],
        world_frame_id: str = "world",
        node_name: str = "isaacscene_camera_publisher",
        publish_clean: bool = False,
        enable_visualization: bool = True,
        visualization_hz: float = 2.0,
        frustum_depth_m: float = 0.45,
        texture_jpeg_quality: int = 75,
        primary_stream_label: str = "clean",
    ) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)

        if visualization_hz <= 0.0:
            raise ValueError("visualization_hz 必须大于 0")
        if frustum_depth_m <= 0.0:
            raise ValueError("frustum_depth_m 必须大于 0")
        if not 1 <= texture_jpeg_quality <= 100:
            raise ValueError("texture_jpeg_quality 必须在 [1, 100]")

        self.node = rclpy.create_node(node_name)
        self.world_frame_id = world_frame_id
        self.publish_clean = publish_clean
        self.enable_visualization = enable_visualization
        self.visualization_period_s = 1.0 / visualization_hz
        self.frustum_depth_m = frustum_depth_m
        self.texture_jpeg_quality = texture_jpeg_quality
        self.primary_stream_label = primary_stream_label.upper()
        self._next_visualization_time = 0.0

        self.cameras = {
            camera.spec.name: camera for camera in cameras
        }

        self.publishers: dict[str, _CameraPublishers] = {}
        for name in self.cameras:
            namespace = f"/{name}"
            publishers = _CameraPublishers(
                color=self.node.create_publisher(
                    Image,
                    f"{namespace}/color/image_raw",
                    qos_profile_sensor_data,
                ),
                depth=self.node.create_publisher(
                    Image,
                    f"{namespace}/depth/image_raw",
                    qos_profile_sensor_data,
                ),
                # 保留旧话题，兼容当前已有脚本。
                camera_info=self.node.create_publisher(
                    CameraInfo,
                    f"{namespace}/camera_info",
                    qos_profile_sensor_data,
                ),
                # 标准 camera transport 命名，RViz Camera Display 会自动寻找。
                color_camera_info=self.node.create_publisher(
                    CameraInfo,
                    f"{namespace}/color/camera_info",
                    qos_profile_sensor_data,
                ),
                points=self.node.create_publisher(
                    PointCloud2,
                    f"{namespace}/points",
                    qos_profile_sensor_data,
                ),
                pose=self.node.create_publisher(
                    PoseStamped,
                    f"{namespace}/pose",
                    10,
                ),
            )

            if publish_clean:
                publishers.clean_color = self.node.create_publisher(
                    Image,
                    f"{namespace}/clean/color/image_raw",
                    qos_profile_sensor_data,
                )
                publishers.clean_depth = self.node.create_publisher(
                    Image,
                    f"{namespace}/clean/depth/image_raw",
                    qos_profile_sensor_data,
                )
                publishers.clean_camera_info = self.node.create_publisher(
                    CameraInfo,
                    f"{namespace}/clean/color/camera_info",
                    qos_profile_sensor_data,
                )
                publishers.clean_points = self.node.create_publisher(
                    PointCloud2,
                    f"{namespace}/clean/points",
                    qos_profile_sensor_data,
                )

            self.publishers[name] = publishers

        self.fused_publisher = self.node.create_publisher(
            PointCloud2,
            "/cameras/fused_points",
            qos_profile_sensor_data,
        )

        self.clean_fused_publisher = None
        if publish_clean:
            self.clean_fused_publisher = self.node.create_publisher(
                PointCloud2,
                "/cameras/clean/fused_points",
                qos_profile_sensor_data,
            )

        self.visualization_publisher = None
        if enable_visualization:
            marker_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.visualization_publisher = self.node.create_publisher(
                MarkerArray,
                "/cameras/visualization",
                marker_qos,
            )

        self.static_tf_broadcaster = StaticTransformBroadcaster(
            self.node
        )
        self._publish_static_transforms()

        self.node.get_logger().info(
            "IsaacScene ROS publisher 已启动；"
            f"主数据流={self.primary_stream_label}；"
            f"3D 可视化={self.enable_visualization}"
        )

    @staticmethod
    def _image_message(
        array: np.ndarray,
        encoding: str,
        frame_id: str,
        stamp,
    ) -> Image:
        contiguous = np.ascontiguousarray(array)

        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = int(contiguous.shape[0])
        message.width = int(contiguous.shape[1])
        message.encoding = encoding
        message.is_bigendian = 0

        if encoding == "rgb8":
            message.step = int(contiguous.shape[1] * 3)
        elif encoding == "32FC1":
            message.step = int(contiguous.shape[1] * 4)
        else:
            raise ValueError(f"不支持的图像编码：{encoding}")

        message.data = contiguous.tobytes()
        return message

    @staticmethod
    def _camera_info_message(
        runtime: CameraRuntime,
        width: int,
        height: int,
        stamp,
    ) -> CameraInfo:
        K = runtime.K

        message = CameraInfo()
        message.header.stamp = stamp
        message.header.frame_id = runtime.spec.optical_frame_id
        message.width = int(width)
        message.height = int(height)
        message.distortion_model = "plumb_bob"
        message.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        message.k = K.reshape(-1).astype(float).tolist()
        message.r = np.eye(
            3, dtype=np.float64
        ).reshape(-1).tolist()
        message.p = [
            float(K[0, 0]), 0.0, float(K[0, 2]), 0.0,
            0.0, float(K[1, 1]), float(K[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return message

    @staticmethod
    def _pointcloud_message(
        points: np.ndarray,
        colors: np.ndarray,
        frame_id: str,
        stamp,
    ) -> PointCloud2:
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"点云尺寸异常：{points.shape}")
        if colors.shape != points.shape:
            raise ValueError(
                f"点和颜色尺寸不一致："
                f"{points.shape} vs {colors.shape}"
            )

        packed_rgb = (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        )

        dtype = np.dtype(
            {
                "names": ["x", "y", "z", "rgb"],
                "formats": ["<f4", "<f4", "<f4", "<u4"],
                "offsets": [0, 4, 8, 12],
                "itemsize": 16,
            }
        )
        cloud = np.empty(points.shape[0], dtype=dtype)
        cloud["x"] = points[:, 0]
        cloud["y"] = points[:, 1]
        cloud["z"] = points[:, 2]
        cloud["rgb"] = packed_rgb

        message = PointCloud2()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = 1
        message.width = int(points.shape[0])
        message.fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="rgb",
                offset=12,
                datatype=PointField.UINT32,
                count=1,
            ),
        ]
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = 16 * int(points.shape[0])
        message.data = cloud.tobytes()
        message.is_dense = bool(np.isfinite(points).all())
        return message

    @staticmethod
    def _pose_message(
        runtime: CameraRuntime,
        world_frame_id: str,
        stamp,
    ) -> PoseStamped:
        transform = runtime.T_world_from_camera_optical
        quaternion = rotation_matrix_to_quaternion_xyzw(
            transform[:3, :3]
        )

        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = world_frame_id
        message.pose.position.x = float(transform[0, 3])
        message.pose.position.y = float(transform[1, 3])
        message.pose.position.z = float(transform[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        return message

    @staticmethod
    def _identity_marker(
        frame_id: str,
        stamp,
        namespace: str,
        marker_id: int,
        marker_type: int,
    ) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = int(marker_type)
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.frame_locked = True
        return marker

    @staticmethod
    def _frustum_corners(
        runtime: CameraRuntime,
        width: int,
        height: int,
        depth_m: float,
    ) -> tuple[Point, Point, Point, Point]:
        K = runtime.K
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        def project(u: float, v: float) -> Point:
            return _point(
                (u - cx) * depth_m / fx,
                (v - cy) * depth_m / fy,
                depth_m,
            )

        top_left = project(0.0, 0.0)
        top_right = project(float(width - 1), 0.0)
        bottom_right = project(
            float(width - 1),
            float(height - 1),
        )
        bottom_left = project(0.0, float(height - 1))
        return top_left, top_right, bottom_right, bottom_left

    def _camera_markers(
        self,
        frame: CameraFrame,
        stamp,
        camera_index: int,
    ) -> list[Marker]:
        runtime = frame.runtime
        optical_frame = runtime.spec.optical_frame_id
        height, width = frame.rgb.shape[:2]
        tl, tr, br, bl = self._frustum_corners(
            runtime,
            width,
            height,
            self.frustum_depth_m,
        )
        origin = _point(0.0, 0.0, 0.0)
        center = _point(0.0, 0.0, self.frustum_depth_m)

        palette = (
            (0.10, 0.85, 1.00),
            (1.00, 0.55, 0.10),
            (0.70, 0.35, 1.00),
            (0.30, 1.00, 0.45),
        )
        camera_color = palette[camera_index % len(palette)]

        # 相机机身。
        body = self._identity_marker(
            optical_frame,
            stamp,
            "camera_body",
            camera_index,
            Marker.CUBE,
        )
        body.pose.position.z = -0.035
        body.scale.x = 0.10
        body.scale.y = 0.065
        body.scale.z = 0.055
        body.color = _color(*camera_color, 0.95)

        # 视锥体线框。
        frustum = self._identity_marker(
            optical_frame,
            stamp,
            "camera_frustum",
            camera_index,
            Marker.LINE_LIST,
        )
        frustum.scale.x = 0.008
        frustum.color = _color(*camera_color, 1.0)
        frustum.points = [
            origin, tl,
            origin, tr,
            origin, br,
            origin, bl,
            tl, tr,
            tr, br,
            br, bl,
            bl, tl,
        ]

        # 光轴箭头。
        optical_axis = self._identity_marker(
            optical_frame,
            stamp,
            "camera_optical_axis",
            camera_index,
            Marker.ARROW,
        )
        optical_axis.points = [origin, center]
        optical_axis.scale.x = 0.014
        optical_axis.scale.y = 0.030
        optical_axis.scale.z = 0.045
        optical_axis.color = _color(0.20, 0.45, 1.00, 1.0)

        # optical frame 坐标轴：x 红、y 绿、z 蓝。
        axis_length = 0.18
        axes = self._identity_marker(
            optical_frame,
            stamp,
            "camera_axes",
            camera_index,
            Marker.LINE_LIST,
        )
        axes.scale.x = 0.012
        axes.points = [
            origin, _point(axis_length, 0.0, 0.0),
            origin, _point(0.0, axis_length, 0.0),
            origin, _point(0.0, 0.0, axis_length),
        ]
        axes.colors = [
            _color(1.0, 0.0, 0.0), _color(1.0, 0.0, 0.0),
            _color(0.0, 1.0, 0.0), _color(0.0, 1.0, 0.0),
            _color(0.0, 0.3, 1.0), _color(0.0, 0.3, 1.0),
        ]

        # 相机名称和当前数据流。
        label = self._identity_marker(
            optical_frame,
            stamp,
            "camera_labels",
            camera_index,
            Marker.TEXT_VIEW_FACING,
        )
        label.pose.position.x = 0.0
        label.pose.position.y = -0.12
        label.pose.position.z = -0.08
        label.scale.z = 0.075
        label.color = _color(1.0, 1.0, 1.0, 1.0)
        label.text = (
            f"{runtime.spec.name} [{self.primary_stream_label}]"
        )

        # 为保证 RViz2 Jazzy 稳定性，这里不再使用嵌入式 JPEG
        # TRIANGLE_LIST 纹理。RGB 图像仍通过标准 Image display 显示。
        return [
            body,
            frustum,
            optical_axis,
            axes,
            label,
        ]

    def _maybe_publish_visualization(
        self,
        frames: dict[str, CameraFrame],
        stamp,
    ) -> None:
        if self.visualization_publisher is None:
            return

        now = time.perf_counter()
        if now < self._next_visualization_time:
            return
        self._next_visualization_time = (
            now + self.visualization_period_s
        )

        marker_array = MarkerArray()
        for camera_index, name in enumerate(sorted(frames)):
            marker_array.markers.extend(
                self._camera_markers(
                    frames[name],
                    stamp,
                    camera_index,
                )
            )

        self.visualization_publisher.publish(marker_array)

    def _publish_static_transforms(self) -> None:
        stamp = self.node.get_clock().now().to_msg()
        transforms: list[TransformStamped] = []

        for runtime in self.cameras.values():
            matrix = runtime.T_world_from_camera_optical
            quaternion = rotation_matrix_to_quaternion_xyzw(
                matrix[:3, :3]
            )

            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.world_frame_id
            transform.child_frame_id = (
                runtime.spec.optical_frame_id
            )
            transform.transform.translation.x = float(
                matrix[0, 3]
            )
            transform.transform.translation.y = float(
                matrix[1, 3]
            )
            transform.transform.translation.z = float(
                matrix[2, 3]
            )
            transform.transform.rotation.x = float(
                quaternion[0]
            )
            transform.transform.rotation.y = float(
                quaternion[1]
            )
            transform.transform.rotation.z = float(
                quaternion[2]
            )
            transform.transform.rotation.w = float(
                quaternion[3]
            )
            transforms.append(transform)

        self.static_tf_broadcaster.sendTransform(transforms)

    def publish(
        self,
        frames: dict[str, CameraFrame],
        fused_points_world: np.ndarray,
        fused_colors: np.ndarray,
        clean_fused_points_world: np.ndarray | None = None,
        clean_fused_colors: np.ndarray | None = None,
    ) -> None:
        stamp = self.node.get_clock().now().to_msg()

        for name, frame in frames.items():
            publishers = self.publishers[name]
            optical_frame = frame.runtime.spec.optical_frame_id

            image_message = self._image_message(
                frame.rgb,
                "rgb8",
                optical_frame,
                stamp,
            )
            depth_message = self._image_message(
                frame.depth_m.astype(np.float32, copy=False),
                "32FC1",
                optical_frame,
                stamp,
            )
            camera_info_message = self._camera_info_message(
                frame.runtime,
                frame.rgb.shape[1],
                frame.rgb.shape[0],
                stamp,
            )

            publishers.color.publish(image_message)
            publishers.depth.publish(depth_message)
            publishers.camera_info.publish(camera_info_message)
            publishers.color_camera_info.publish(
                camera_info_message
            )
            publishers.points.publish(
                self._pointcloud_message(
                    frame.points_camera_optical,
                    frame.colors,
                    optical_frame,
                    stamp,
                )
            )
            publishers.pose.publish(
                self._pose_message(
                    frame.runtime,
                    self.world_frame_id,
                    stamp,
                )
            )

            if self.publish_clean:
                assert publishers.clean_color is not None
                assert publishers.clean_depth is not None
                assert publishers.clean_camera_info is not None
                assert publishers.clean_points is not None

                publishers.clean_color.publish(
                    self._image_message(
                        frame.clean_rgb,
                        "rgb8",
                        optical_frame,
                        stamp,
                    )
                )
                publishers.clean_depth.publish(
                    self._image_message(
                        frame.clean_depth_m.astype(
                            np.float32, copy=False
                        ),
                        "32FC1",
                        optical_frame,
                        stamp,
                    )
                )
                publishers.clean_camera_info.publish(
                    camera_info_message
                )
                publishers.clean_points.publish(
                    self._pointcloud_message(
                        frame.clean_points_camera_optical,
                        frame.clean_colors,
                        optical_frame,
                        stamp,
                    )
                )

        self.fused_publisher.publish(
            self._pointcloud_message(
                fused_points_world,
                fused_colors,
                self.world_frame_id,
                stamp,
            )
        )

        if (
            self.publish_clean
            and self.clean_fused_publisher is not None
            and clean_fused_points_world is not None
            and clean_fused_colors is not None
        ):
            self.clean_fused_publisher.publish(
                self._pointcloud_message(
                    clean_fused_points_world,
                    clean_fused_colors,
                    self.world_frame_id,
                    stamp,
                )
            )

        self._maybe_publish_visualization(frames, stamp)

        # 处理 DDS 事件，不阻塞 Isaac Sim。
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self) -> None:
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()