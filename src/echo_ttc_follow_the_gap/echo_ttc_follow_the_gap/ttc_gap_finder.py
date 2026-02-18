#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class TTCGapFinder(Node):

    def __init__(self):
        super().__init__('ttc_gap_finder')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('ttc_min', 0.5)
        self.declare_parameter('v_forward', 1.0)

        self.ttc_min = self.get_parameter('ttc_min').value
        self.v = self.get_parameter('v_forward').value

        # ---------------- SUBSCRIBER ----------------
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # ---------------- PUBLISHERS ----------------
        self.gap_pub = self.create_publisher(Float32, '/gap_angle', 10)
        self.ttc_pub = self.create_publisher(Float32, '/min_ttc', 10)

        self.get_logger().info("TTC Gap Finder Started")

    # --------------------------------------------------
    def compute_ttc(self, distance, angle):
        """Compute TTC for a single LiDAR ray"""

        relative_velocity = self.v * math.cos(angle)

        if relative_velocity <= 0.0:
            return float('inf')

        return distance / relative_velocity

    # --------------------------------------------------
    def scan_callback(self, scan):

        ranges = scan.ranges
        angle = scan.angle_min

        safe_mask = []
        min_ttc = float('inf')

        # ---------- Compute TTC for each ray ----------
        for r in ranges:

            if math.isinf(r) or math.isnan(r):
                safe_mask.append(True)
                angle += scan.angle_increment
                continue

            ttc = self.compute_ttc(r, angle)

            min_ttc = min(min_ttc, ttc)

            safe_mask.append(ttc >= self.ttc_min)

            angle += scan.angle_increment

        # ---------- Find largest contiguous safe gap ----------
        max_gap_start = 0
        max_gap_size = 0

        current_start = None
        current_size = 0

        for i, safe in enumerate(safe_mask):

            if safe:
                if current_start is None:
                    current_start = i
                    current_size = 1
                else:
                    current_size += 1
            else:
                if current_size > max_gap_size:
                    max_gap_size = current_size
                    max_gap_start = current_start
                current_start = None
                current_size = 0

        # Final check
        if current_size > max_gap_size:
            max_gap_size = current_size
            max_gap_start = current_start

        # ---------- Compute center angle ----------
        if max_gap_size > 0:
            center_index = max_gap_start + max_gap_size // 2
            gap_angle = scan.angle_min + center_index * scan.angle_increment
        else:
            gap_angle = 0.0  # fallback

        # ---------- Publish ----------
        gap_msg = Float32()
        gap_msg.data = float(gap_angle)
        self.gap_pub.publish(gap_msg)

        ttc_msg = Float32()
        ttc_msg.data = float(min_ttc)
        self.ttc_pub.publish(ttc_msg)

        self.get_logger().info(
            f"Gap angle: {math.degrees(gap_angle):.2f} deg | Min TTC: {min_ttc:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TTCGapFinder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
