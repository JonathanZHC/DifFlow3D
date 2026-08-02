#!/usr/bin/env python3
"""Measure the actual rclpy receive rate of a sensor_msgs/Image topic.

The callback does not convert, copy, display, or process image pixels. It only
counts callbacks and reads metadata that has already been deserialized by
rclpy.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        default="/camera_0/depth/image_raw",
    )
    parser.add_argument(
        "--reliability",
        choices=("best_effort", "reliable"),
        default="best_effort",
    )
    parser.add_argument(
        "--expected-hz",
        type=float,
        default=30.0,
    )
    return parser.parse_args()


class ImageRateSubscriber(Node):
    def __init__(
        self,
        topic: str,
        reliability: str,
        expected_hz: float,
    ) -> None:
        super().__init__("python_image_rate_subscriber")

        if reliability == "reliable":
            reliability_policy = ReliabilityPolicy.RELIABLE
        else:
            reliability_policy = ReliabilityPolicy.BEST_EFFORT

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=reliability_policy,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.expected_period = 1.0 / expected_hz
        self.window_start = time.perf_counter()
        self.total_start = self.window_start
        self.window_count = 0
        self.total_count = 0
        self.last_receive_time: float | None = None
        self.last_source_stamp: float | None = None
        self.inter_arrivals: list[float] = []
        self.source_stamp_gaps: list[float] = []
        self.estimated_missing = 0
        self.last_payload_bytes = 0
        self.last_encoding = ""

        self.subscription = self.create_subscription(
            Image,
            topic,
            self.callback,
            qos,
        )
        self.report_timer = self.create_timer(
            1.0,
            self.report,
        )

        self.get_logger().info(
            f"Subscribing to {topic}, "
            f"reliability={reliability}, "
            "KEEP_LAST(1)."
        )

    def callback(self, message: Image) -> None:
        receive_time = time.perf_counter()
        source_stamp = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )

        if self.last_receive_time is not None:
            self.inter_arrivals.append(
                receive_time - self.last_receive_time
            )

        if (
            self.last_source_stamp is not None
            and source_stamp > self.last_source_stamp
        ):
            source_gap = source_stamp - self.last_source_stamp
            self.source_stamp_gaps.append(source_gap)

            frame_steps = max(
                1,
                int(round(source_gap / self.expected_period)),
            )
            self.estimated_missing += max(0, frame_steps - 1)

        self.last_receive_time = receive_time
        self.last_source_stamp = source_stamp
        self.window_count += 1
        self.total_count += 1
        self.last_payload_bytes = len(message.data)
        self.last_encoding = message.encoding

    def report(self) -> None:
        now = time.perf_counter()
        window_seconds = now - self.window_start
        total_seconds = now - self.total_start

        receive_hz = (
            self.window_count / window_seconds
            if window_seconds > 0.0
            else 0.0
        )
        average_hz = (
            self.total_count / total_seconds
            if total_seconds > 0.0
            else 0.0
        )

        mean_arrival_ms = (
            statistics.fmean(self.inter_arrivals) * 1000.0
            if self.inter_arrivals
            else math.nan
        )
        max_arrival_ms = (
            max(self.inter_arrivals) * 1000.0
            if self.inter_arrivals
            else math.nan
        )
        max_source_gap_ms = (
            max(self.source_stamp_gaps) * 1000.0
            if self.source_stamp_gaps
            else math.nan
        )

        print(
            f"received_hz={receive_hz:7.3f} "
            f"average_hz={average_hz:7.3f} "
            f"messages={self.window_count:3d} "
            f"mean_arrival_ms={mean_arrival_ms:8.3f} "
            f"max_arrival_ms={max_arrival_ms:8.3f} "
            f"max_source_gap_ms={max_source_gap_ms:8.3f} "
            f"estimated_missing={self.estimated_missing:5d} "
            f"bytes={self.last_payload_bytes} "
            f"encoding={self.last_encoding}",
            flush=True,
        )

        self.window_start = now
        self.window_count = 0
        self.inter_arrivals.clear()
        self.source_stamp_gaps.clear()
        self.estimated_missing = 0


def main() -> None:
    arguments = parse_arguments()
    rclpy.init()
    node = ImageRateSubscriber(
        topic=arguments.topic,
        reliability=arguments.reliability,
        expected_hz=arguments.expected_hz,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
