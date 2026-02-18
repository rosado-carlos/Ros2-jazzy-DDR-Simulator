#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped


class TTCControl(Node):

    def __init__(self):
        super().__init__('ttc_control')

        # -------- PARAMETERS --------
        self.declare_parameter('kp_steering', 3.0)
        self.declare_parameter('v_max', 3.0)
        self.declare_parameter('ttc_min', 0.8)
        self.declare_parameter('max_steering', 2.5)
        self.declare_parameter('steering_slowdown_gain', 2.0)

        self.kp = self.get_parameter('kp_steering').value
        self.v_max = self.get_parameter('v_max').value
        self.ttc_min = self.get_parameter('ttc_min').value
        self.max_steering = self.get_parameter('max_steering').value
        self.steering_slowdown_gain = self.get_parameter(
            'steering_slowdown_gain').value

        self.gap_angle = 0.0
        self.min_ttc = float('inf')

        self.create_subscription(Float32, '/gap_angle', self.gap_callback, 10)
        self.create_subscription(Float32, '/min_ttc', self.ttc_callback, 10)

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel_gap',
            10
        )

        self.timer = self.create_timer(0.05, self.compute_and_publish)

    # --------------------------------------------------
    def gap_callback(self, msg):
        self.gap_angle = msg.data

    # --------------------------------------------------
    def ttc_callback(self, msg):
        self.min_ttc = msg.data

    # --------------------------------------------------
    def compute_and_publish(self):

        # -------- STEERING --------
        steering = self.kp * self.gap_angle
        steering = max(-self.max_steering,
                       min(self.max_steering, steering))

        # -------- BASE VELOCITY FROM TTC --------
        if self.min_ttc < self.ttc_min:
            base_velocity = 0.1
        else:
            base_velocity = min(
                self.v_max,
                self.v_max * (self.min_ttc / 3.0)
            )

        # -------- SLOW DOWN WHEN TURNING --------
        velocity = base_velocity / (
            1.0 + self.steering_slowdown_gain * abs(steering)
        )

        # -------- PUBLISH --------
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"

        cmd.twist.linear.x = velocity
        cmd.twist.angular.z = steering

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TTCControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()