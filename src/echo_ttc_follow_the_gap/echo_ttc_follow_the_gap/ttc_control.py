#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped


class TTCControl(Node):

    def __init__(self):
        super().__init__('ttc_control')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('kp_steering', 1.5)
        self.declare_parameter('v_max', 2.0)
        self.declare_parameter('ttc_min', 0.5)
        self.declare_parameter('max_steering', 1.0)

        self.kp = self.get_parameter('kp_steering').value
        self.v_max = self.get_parameter('v_max').value
        self.ttc_min = self.get_parameter('ttc_min').value
        self.max_steering = self.get_parameter('max_steering').value

        # ---------------- STATE ----------------
        self.gap_angle = 0.0
        self.min_ttc = float('inf')

        # ---------------- SUBSCRIBERS ----------------
        self.gap_sub = self.create_subscription(
            Float32,
            '/gap_angle',
            self.gap_callback,
            10
        )

        self.ttc_sub = self.create_subscription(
            Float32,
            '/min_ttc',
            self.ttc_callback,
            10
        )

        # ---------------- PUBLISHER ----------------
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel_gap',
            10
        )

        self.get_logger().info("TTC Control Node Started")

    # --------------------------------------------------
    def gap_callback(self, msg):
        self.gap_angle = msg.data
        self.compute_and_publish()

    # --------------------------------------------------
    def ttc_callback(self, msg):
        self.min_ttc = msg.data

    # --------------------------------------------------
    def compute_and_publish(self):

        # ---------- Steering Control ----------
        steering = self.kp * self.gap_angle

        # Saturation
        steering = max(-self.max_steering,
                       min(self.max_steering, steering))

        # ---------- Velocity Control ----------
        if self.min_ttc < self.ttc_min:
            velocity = 0.0
        else:
            scale = self.min_ttc / (2.0 * self.ttc_min)
            scale = max(0.1, min(1.0, scale))
            velocity = self.v_max * scale

        # ---------- Publish ----------
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"

        cmd.twist.linear.x = velocity
        cmd.twist.angular.z = steering

        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"Steer: {math.degrees(self.gap_angle):.1f}° | "
            f"v: {velocity:.2f} | TTC: {self.min_ttc:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TTCControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
