"""ROS 2 / RViz benchmark publisher."""
import time
import numpy as np
import torch

from difflow3d.runtime import DenseMotionRecovery, DifFlow3DEstimate
from .first_voxel import FirstVoxelFrame

class RvizPipelinePublisher:
    def __init__(
        self,
        *,
        frame_id: str,
        max_arrows: int,
        vector_scale: float,
        cloud_max_points: int,
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Point
            from rclpy.node import Node
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import PointCloud2
            from sensor_msgs_py import point_cloud2
            from std_msgs.msg import Header
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as error:
            raise RuntimeError("RViz dependencies are not available.") from error
        self._rclpy = rclpy
        self._Point = Point
        self._point_cloud2 = point_cloud2
        self._Header = Header
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self.frame_id = frame_id
        self.max_arrows = int(max_arrows)
        self.vector_scale = float(vector_scale)
        self.cloud_max_points = int(cloud_max_points)
        if not rclpy.ok():
            rclpy.init(args=[])
        self.node: Node = rclpy.create_node("difflow3d_integrated_runner_visualizer")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.raw_pub = self.node.create_publisher(PointCloud2, "/pipeline/raw_target", qos)
        self.first_target_pub = self.node.create_publisher(PointCloud2, "/pipeline/first_downsample_target", qos)
        self.anchor_source_pub = self.node.create_publisher(PointCloud2, "/pipeline/anchor_source", qos)
        self.anchor_target_pub = self.node.create_publisher(PointCloud2, "/pipeline/anchor_target", qos)
        self.anchor_warped_pub = self.node.create_publisher(PointCloud2, "/pipeline/anchor_predicted_warped", qos)
        self.first_source_pub = self.node.create_publisher(PointCloud2, "/pipeline/recovered_first_source", qos)
        self.first_warped_pub = self.node.create_publisher(PointCloud2, "/pipeline/recovered_first_warped", qos)
        self.anchor_vectors_pub = self.node.create_publisher(MarkerArray, "/pipeline/anchor_displacement_vectors", qos)
        self.first_vectors_pub = self.node.create_publisher(MarkerArray, "/pipeline/recovered_first_displacement_vectors", qos)

    def _header(self):
        header = self._Header()
        header.frame_id = self.frame_id
        header.stamp = self.node.get_clock().now().to_msg()
        return header

    def _cloud(self, points: np.ndarray):
        xyz = np.ascontiguousarray(points, dtype=np.float32)
        return self._point_cloud2.create_cloud_xyz32(self._header(), xyz)

    @staticmethod
    def _uniform_indices(count: int, limit: int) -> np.ndarray:
        if limit <= 0 or count <= limit:
            return np.arange(count, dtype=np.int64)
        return np.linspace(0, count - 1, limit, dtype=np.int64)

    def _flow_marker(self, marker_id, namespace, source_points, flow, rgb, line_width):
        marker = self._Marker()
        marker.header = self._header()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = self._Marker.LINE_LIST
        marker.action = self._Marker.ADD
        marker.scale.x = float(line_width)
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 0.9
        marker.pose.orientation.w = 1.0
        for index in self._uniform_indices(source_points.shape[0], self.max_arrows):
            start = source_points[index]
            end = start + self.vector_scale * flow[index]
            a = self._Point(); a.x, a.y, a.z = map(float, start)
            b = self._Point(); b.x, b.y, b.z = map(float, end)
            marker.points.extend((a, b))
        return marker

    def publish_buffered(self, frame: FirstVoxelFrame) -> None:
        raw_idx = self._uniform_indices(frame.raw_count, self.cloud_max_points)
        first = frame.first_downsample_points.detach().cpu().numpy()
        first_idx = self._uniform_indices(first.shape[0], self.cloud_max_points)
        self.raw_pub.publish(self._cloud(frame.raw_points_cpu[raw_idx]))
        self.first_target_pub.publish(self._cloud(first[first_idx]))
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_pair(
        self,
        *,
        source_frame: FirstVoxelFrame,
        target_frame: FirstVoxelFrame,
        estimate: DifFlow3DEstimate,
        recovery: DenseMotionRecovery,
        anchor_flow_override: torch.Tensor | None = None,
        anchor_gt_flow: np.ndarray,
        first_gt_flow: np.ndarray,
    ) -> None:
        self.publish_buffered(target_frame)
        anchor_source = estimate.source_points.detach().cpu().numpy()
        anchor_target = estimate.target_points.detach().cpu().numpy()
        anchor_flow = (
            estimate.residual_flow
            if anchor_flow_override is None
            else anchor_flow_override
        ).detach().cpu().numpy()
        # Keep the displayed warped cloud consistent with the arrows.  When a
        # filtered flow override is supplied, ``estimate.warped_points`` is
        # still the raw DifFlow warp and would otherwise make RViz show a
        # misleading mixture of raw and filtered results.
        anchor_warped = anchor_source + anchor_flow
        first_source = source_frame.first_downsample_points.detach().cpu().numpy()
        first_flow = recovery.flow.detach().cpu().numpy()
        first_warped = first_source + first_flow
        self.anchor_source_pub.publish(self._cloud(anchor_source))
        self.anchor_target_pub.publish(self._cloud(anchor_target))
        self.anchor_warped_pub.publish(self._cloud(anchor_warped))
        self.first_source_pub.publish(self._cloud(first_source))
        self.first_warped_pub.publish(self._cloud(first_warped))
        am = self._MarkerArray()
        am.markers.append(self._flow_marker(0, "anchor_predicted", anchor_source, anchor_flow, (1.0,0.15,0.15), 0.004))
        am.markers.append(self._flow_marker(1, "anchor_ground_truth", anchor_source, anchor_gt_flow, (0.15,1.0,0.20), 0.0025))
        self.anchor_vectors_pub.publish(am)
        fm = self._MarkerArray()
        fm.markers.append(self._flow_marker(0, "first_recovered", first_source, first_flow, (0.20,0.45,1.0), 0.003))
        fm.markers.append(self._flow_marker(1, "first_ground_truth", first_source, first_gt_flow, (0.15,1.0,0.20), 0.002))
        self.first_vectors_pub.publish(fm)
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def hold(self, seconds: float) -> None:
        if seconds == 0.0:
            return
        if seconds < 0.0:
            try:
                while self._rclpy.ok():
                    self._rclpy.spin_once(self.node, timeout_sec=0.1)
            except KeyboardInterrupt:
                pass
            return
        deadline = time.perf_counter() + seconds
        while self._rclpy.ok() and time.perf_counter() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
