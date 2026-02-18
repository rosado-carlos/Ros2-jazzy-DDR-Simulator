#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped
import time


class WallFollowerControl(Node):

    def __init__(self):
        super().__init__('wall_follower_control')

        # -------- Parameters --------
        self.declare_parameter('kp', 1.1)
        self.declare_parameter('kd', 1.5)
        self.declare_parameter('max_steering', 100)

        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('min_velocity', 0.2)
        self.declare_parameter('kv', 3.0)  # velocidad adaptativa

        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.max_steering = self.get_parameter('max_steering').value

        self.max_velocity = self.get_parameter('max_velocity').value
        self.min_velocity = self.get_parameter('min_velocity').value
        self.kv = self.get_parameter('kv').value

        # -------- State --------
        self.prev_error = 0.0
        self.prev_time = time.time()

        # -------- Subscriber --------
        self.error_sub = self.create_subscription(
            Float32,
            '/error',
            self.error_callback,
            10
        )

        # -------- Publisher --------
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel_ctrl',
            10
        )

        self.get_logger().info("Wall follower control node started")

    # ------------------------------------------------------
    def error_callback(self, msg):

        current_time = time.time()
        dt = current_time - self.prev_time

        if dt <= 0.0:
            return

        error = msg.data

        # -------- PD Steering --------
        derivative = (error - self.prev_error) / dt
        steering = self.kp * error + self.kd * derivative

        # -------- Steering Saturation --------
        if steering > self.max_steering:
            steering = self.max_steering
        elif steering < -self.max_steering:
            steering = -self.max_steering

        # -------- Adaptive Velocity --------
        velocity = self.max_velocity / (1 + self.kv * abs(error))

        if velocity < self.min_velocity:
            velocity = self.min_velocity

        # -------- Construct TwistStamped --------
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"

        cmd.twist.linear.x = velocity
        cmd.twist.angular.z = steering

        # -------- Publish --------
        self.cmd_pub.publish(cmd)

        # -------- Update state --------
        self.prev_error = error
        self.prev_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowerControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
