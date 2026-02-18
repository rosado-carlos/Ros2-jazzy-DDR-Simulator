#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class DistFinder(Node):

    def __init__(self):
        super().__init__('dist_finder')

        # --- Parameters ---
        self.declare_parameter('theta_deg', 53.0)
        self.declare_parameter('lookahead_dist', 1.5)
        self.declare_parameter('desired_distance', 0.85)

        self.theta = math.radians(
            self.get_parameter('theta_deg').value 
        ) 
        self.L = self.get_parameter('lookahead_dist').value
        self.desired_distance = self.get_parameter('desired_distance').value

        # --- ROS Interfaces ---
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.error_pub = self.create_publisher(
            Float32,
            '/error',
            10
        )

        self.get_logger().info("dist_finder started")

    # --------------------------------------------------
    def getRange(self, data, angle):

        index = int((angle - data.angle_min) / data.angle_increment)

        index = max(0, min(index, len(data.ranges) - 1))

        r = data.ranges[index]

        if math.isinf(r) or math.isnan(r):
            return data.range_max

        return r

    # --------------------------------------------------
    def scan_callback(self, data):

        # Rays
        b = self.getRange(data, - math.pi/2)          # distance at 0°
        a = self.getRange(data, -math.pi/2 + self.theta)  # distance at θ°

        if a == 0.0 or b == 0.0:
            return

        # --- Wall orientation angle θ (called alpha) ---
        alpha = math.atan2(
            a * math.cos(self.theta) - b,
            a * math.sin(self.theta)
        )

        # --- Current lateral distance y ---
        y = b * math.cos(alpha)

        # --- Future projected distance y + L sin(θ) ---
        y_future = y + self.L * math.sin(alpha)

        # --- Error definition ---
        error = self.desired_distance - y_future

        msg = Float32()
        msg.data = float(error)
        self.error_pub.publish(msg)
        self.get_logger().info(f"Error: {error: .3f}, Alpha: {math.degrees(alpha): .3f}")


def main(args=None):
    rclpy.init(args=args)
    node = DistFinder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()