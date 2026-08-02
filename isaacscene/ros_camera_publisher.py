#!/usr/bin/env python3
"""Publish two independent RGB-D cameras and full per-camera point clouds.

RGB, 32FC1 depth, CameraInfo and pose are published at the RGB-D rate.
Each camera also publishes its own full valid XYZRGB PointCloud2 at an
independent lower rate.

Point clouds remain in their camera optical frames. No downsampling, camera
fusion, world-frame transformation or point-count limiting is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import (
        Point,
        PoseStamped,
        TransformStamped,
    )
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
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
        "ROS 2 Python camera dependencies are missing."
    ) from error

from camera_settings import (
    CameraFrame,
    CameraRuntime,
    rotation_matrix_to_quaternion_xyzw,
)
from pointcloud_full_gpu import GpuCameraPointCloudBuilder


PYRAMID_DEPTH_M = 0.45
POINTCLOUD_DEVICE = "cuda:0"
DEPTH_ENCODING = "32FC1"


@dataclass
class _CameraPublishers:
    color: Any
    depth: Any
    camera_info: Any
    pose: Any
    points: Any


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
    """Publish independent camera topics without point-cloud fusion."""

    def __init__(
        self,
        cameras: list[CameraRuntime],
        rgbd_hz: float,
        pointcloud_hz: float,
        max_depth_m: float,
        world_frame_id: str = "world",
    ) -> None:
        if rgbd_hz <= 0.0:
            raise ValueError("rgbd_hz must be positive.")
        if pointcloud_hz <= 0.0:
            raise ValueError("pointcloud_hz must be positive.")
        if pointcloud_hz > rgbd_hz:
            raise ValueError(
                "pointcloud_hz cannot exceed rgbd_hz because point clouds "
                "are generated from captured RGB-D frames."
            )

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node(
            "isaacscene_camera_publisher"
        )
        self.world_frame_id = world_frame_id
        self.rgbd_hz = float(rgbd_hz)
        self.pointcloud_hz = float(pointcloud_hz)

        self._pointcloud_period_s = 1.0 / self.pointcloud_hz
        self._next_pointcloud_time = time.perf_counter()

        self._pointcloud_builder = GpuCameraPointCloudBuilder(
            max_depth_m=max_depth_m,
            device=POINTCLOUD_DEVICE,
        )

        self.last_point_counts: dict[str, int] = {
            camera.spec.name: 0 for camera in cameras
        }
        self.last_pointcloud_ms = 0.0

        # RGB and depth are perception inputs, so use a short reliable queue.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Full point clouds are lower-rate perception inputs.
        pointcloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        metadata_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.cameras = {
            camera.spec.name: camera for camera in cameras
        }

        self.publishers: dict[str, _CameraPublishers] = {}
        for name in self.cameras:
            namespace = f"/{name}"
            self.publishers[name] = _CameraPublishers(
                color=self.node.create_publisher(
                    Image,
                    f"{namespace}/color/image_raw",
                    image_qos,
                ),
                depth=self.node.create_publisher(
                    Image,
                    f"{namespace}/depth/image_raw",
                    image_qos,
                ),
                camera_info=self.node.create_publisher(
                    CameraInfo,
                    f"{namespace}/camera_info",
                    metadata_qos,
                ),
                pose=self.node.create_publisher(
                    PoseStamped,
                    f"{namespace}/pose",
                    metadata_qos,
                ),
                points=self.node.create_publisher(
                    PointCloud2,
                    f"{namespace}/points",
                    pointcloud_qos,
                ),
            )

        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.visualization_publisher = (
            self.node.create_publisher(
                MarkerArray,
                "/cameras/visualization",
                marker_qos,
            )
        )

        self.static_tf_broadcaster = StaticTransformBroadcaster(
            self.node
        )
        self._publish_static_transforms()
        self._publish_camera_pyramids()

        self.node.get_logger().info(
            f"Publishing RGB8 and {DEPTH_ENCODING} at "
            f"{self.rgbd_hz:.3f} Hz. Publishing separate, full, "
            f"non-downsampled optical-frame PointCloud2 streams at "
            f"{self.pointcloud_hz:.3f} Hz."
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
            raise ValueError(
                f"Unsupported image encoding: {encoding}."
            )

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
        message.d = [0.0] * 5
        message.k = K.reshape(-1).astype(float).tolist()
        message.r = np.eye(
            3,
            dtype=np.float64,
        ).reshape(-1).tolist()
        message.p = [
            float(K[0, 0]), 0.0, float(K[0, 2]), 0.0,
            0.0, float(K[1, 1]), float(K[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return message

    @staticmethod
    def _pointcloud_message(
        cloud_xyzirgb: np.ndarray,
        frame_id: str,
        stamp,
    ) -> PointCloud2:
        """Build PointCloud2 from an Nx4 float32 XYZ plus packed-RGB array."""

        cloud = np.ascontiguousarray(
            cloud_xyzirgb,
            dtype=np.float32,
        )
        if cloud.ndim != 2 or cloud.shape[1] != 4:
            raise ValueError(
                f"Point cloud shape is {cloud.shape}, expected Nx4."
            )

        message = PointCloud2()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = 1
        message.width = int(cloud.shape[0])
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
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = 16 * int(cloud.shape[0])
        message.data = cloud.tobytes()
        message.is_dense = True
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
    def _frustum_corners(
        runtime: CameraRuntime,
        width: int,
        height: int,
    ) -> tuple[Point, Point, Point, Point]:
        K = runtime.K
        depth = PYRAMID_DEPTH_M

        def project(u: float, v: float) -> Point:
            return _point(
                (u - float(K[0, 2]))
                * depth / float(K[0, 0]),
                (v - float(K[1, 2]))
                * depth / float(K[1, 1]),
                depth,
            )

        return (
            project(0.0, 0.0),
            project(float(width - 1), 0.0),
            project(float(width - 1), float(height - 1)),
            project(0.0, float(height - 1)),
        )

    def _publish_static_transforms(self) -> None:
        stamp = self.node.get_clock().now().to_msg()
        messages: list[TransformStamped] = []

        for runtime in self.cameras.values():
            transform = runtime.T_world_from_camera_optical
            quaternion = rotation_matrix_to_quaternion_xyzw(
                transform[:3, :3]
            )

            message = TransformStamped()
            message.header.stamp = stamp
            message.header.frame_id = self.world_frame_id
            message.child_frame_id = (
                runtime.spec.optical_frame_id
            )
            message.transform.translation.x = float(
                transform[0, 3]
            )
            message.transform.translation.y = float(
                transform[1, 3]
            )
            message.transform.translation.z = float(
                transform[2, 3]
            )
            message.transform.rotation.x = float(quaternion[0])
            message.transform.rotation.y = float(quaternion[1])
            message.transform.rotation.z = float(quaternion[2])
            message.transform.rotation.w = float(quaternion[3])
            messages.append(message)

        self.static_tf_broadcaster.sendTransform(messages)

    def _publish_camera_pyramids(self) -> None:
        stamp = self.node.get_clock().now().to_msg()
        palette = (
            (0.10, 0.85, 1.00),
            (1.00, 0.55, 0.10),
        )
        marker_array = MarkerArray()

        for index, runtime in enumerate(
            self.cameras.values()
        ):
            tl, tr, br, bl = self._frustum_corners(
                runtime,
                width=int(runtime.ray_x.shape[1]),
                height=int(runtime.ray_x.shape[0]),
            )
            origin = _point(0.0, 0.0, 0.0)

            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = (
                runtime.spec.optical_frame_id
            )
            marker.ns = "camera_pyramids"
            marker.id = index
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.008
            marker.color = _color(
                *palette[index % len(palette)],
                1.0,
            )
            marker.frame_locked = True
            marker.points = [
                origin, tl,
                origin, tr,
                origin, br,
                origin, bl,
                tl, tr,
                tr, br,
                br, bl,
                bl, tl,
            ]
            marker_array.markers.append(marker)

        self.visualization_publisher.publish(marker_array)

    def publish(
        self,
        frames: dict[str, CameraFrame],
    ) -> float:
        """Publish one synchronized RGB-D sample and optional point clouds."""

        stamp = self.node.get_clock().now().to_msg()
        now = time.perf_counter()

        publish_pointcloud = now >= self._next_pointcloud_time
        if publish_pointcloud:
            self._next_pointcloud_time = max(
                self._next_pointcloud_time
                + self._pointcloud_period_s,
                now,
            )

        pointcloud_start = (
            time.perf_counter()
            if publish_pointcloud
            else 0.0
        )

        for name, frame in frames.items():
            publishers = self.publishers[name]
            frame_id = frame.runtime.spec.optical_frame_id

            publishers.color.publish(
                self._image_message(
                    frame.rgb,
                    "rgb8",
                    frame_id,
                    stamp,
                )
            )
            publishers.depth.publish(
                self._image_message(
                    frame.depth_m,
                    DEPTH_ENCODING,
                    frame_id,
                    stamp,
                )
            )
            publishers.camera_info.publish(
                self._camera_info_message(
                    frame.runtime,
                    frame.rgb.shape[1],
                    frame.rgb.shape[0],
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

            if publish_pointcloud:
                cloud = self._pointcloud_builder.build(frame)
                self.last_point_counts[name] = int(cloud.shape[0])
                publishers.points.publish(
                    self._pointcloud_message(
                        cloud,
                        frame_id,
                        stamp,
                    )
                )

        if publish_pointcloud:
            pointcloud_ms = 1000.0 * (
                time.perf_counter() - pointcloud_start
            )
            self.last_pointcloud_ms = pointcloud_ms
        else:
            pointcloud_ms = 0.0

        rclpy.spin_once(self.node, timeout_sec=0.0)
        return pointcloud_ms

    def shutdown(self) -> None:
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
