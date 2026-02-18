#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class TTCGapFinder(Node):

    def __init__(self):
        super().__init__('ttc_gap_finder')

        self.declare_parameter('bubble_radius', 1.5)
        self.declare_parameter('fov_deg', 100.0)
        self.declare_parameter('v_forward', 0.5)

        self.bubble_radius = self.get_parameter('bubble_radius').value
        self.fov = math.radians(self.get_parameter('fov_deg').value)
        self.v = self.get_parameter('v_forward').value

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.gap_pub = self.create_publisher(Float32, '/gap_angle', 10)
        self.ttc_pub = self.create_publisher(Float32, '/min_ttc', 10)

    # --------------------------------------------------
    def compute_ttc(self, distance, angle):
        relative_velocity = self.v * math.cos(angle)

        if relative_velocity <= 0.0:
            return float('inf')

        return distance / relative_velocity

    # --------------------------------------------------
    def scan_callback(self, scan):

        ranges = list(scan.ranges)
        min_ttc = float('inf')

        # -------- LIMIT FOV --------
        for i in range(len(ranges)):
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) > self.fov:
                ranges[i] = 0.0

        # -------- FIND CLOSEST POINT --------
        min_distance = float('inf')
        closest_index = 0

        for i, r in enumerate(ranges):
            if r > 0.0 and r < min_distance:
                min_distance = r
                closest_index = i

        # -------- CREATE BUBBLE --------
        if min_distance < float('inf'):

            bubble_angle = math.atan2(self.bubble_radius, min_distance)
            bubble_size = int(bubble_angle / scan.angle_increment)

            start = max(0, closest_index - bubble_size)
            end = min(len(ranges), closest_index + bubble_size)

            for i in range(start, end):
                ranges[i] = 0.0

        # -------- FIND LARGEST GAP --------
        max_gap_start = 0
        max_gap_size = 0
        current_start = None
        current_size = 0

        for i, r in enumerate(ranges):

            if r > 0.0:
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

        if current_size > max_gap_size:
            max_gap_size = current_size
            max_gap_start = current_start

        # -------- SELECT FARTHEST POINT --------
        # -------- SELECT CENTER OF GAP --------
        if max_gap_size > 0:

            best_index = max_gap_start + max_gap_size // 2
            gap_angle = scan.angle_min + best_index * scan.angle_increment


        else:
            gap_angle = 0.0

        # -------- COMPUTE MIN TTC --------
        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            if r > 0.0:
                ttc = self.compute_ttc(r, angle)
                min_ttc = min(min_ttc, ttc)

        # -------- PUBLISH --------
        gap_msg = Float32()
        gap_msg.data = float(gap_angle)
        self.gap_pub.publish(gap_msg)

        ttc_msg = Float32()
        ttc_msg.data = float(min_ttc)
        self.ttc_pub.publish(ttc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TTCGapFinder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()