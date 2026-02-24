#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped


class AEBNode(Node):

    def __init__(self):
        super().__init__('aeb')

        # -------- Parameters --------
        self.declare_parameter("ttc_threshold", 1.0)
        self.declare_parameter("min_distance", 0.55)
        self.declare_parameter("robot_radius", 0.3)

        self.ttc_threshold = float(self.get_parameter("ttc_threshold").value)
        self.min_distance = float(self.get_parameter("min_distance").value)
        self.radius = float(self.get_parameter("robot_radius").value)

        # -------- State --------
        self.vx = 0.0
        self.d_min = float("inf")
        self.lock = False
        self.forward_Block = False
        self.twist_w = 0.0

        # -------- Subscriptions --------
        self.create_subscription(Float32, "/lidar/vx", self.vx_callback, 1)
        self.create_subscription(Float32, "/lidar/front_p10", self.front_callback, 1)

        self.create_subscription(TwistStamped, "/cmd_vel_joy", self.cmd_callback, 1)
        self.create_subscription(TwistStamped, "/cmd_vel_ctrl", self.cmd_callback, 1)

        # -------- Publisher --------
        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel_safe", 10)

        self.get_logger().info("AEB started")

    # --------------------------------------------------
    def vx_callback(self, msg):
        self.vx = float(msg.data)
        self.compute()

    def front_callback(self, msg):
        self.d_min = float(msg.data)
        self.compute()

    # --------------------------------------------------
    def compute(self):

        if self.vx <= 0.0:
            return

        if not math.isfinite(self.d_min):
            return

        dist = self.d_min - self.radius

        if dist <= 0.0:
            ttc_min = 0.0
        else:
            ttc_min = dist / self.vx

        prev_lock = self.lock
        self.lock = (self.d_min <= 1.5) and (ttc_min < self.ttc_threshold)

        if self.lock and not prev_lock:
            self.get_logger().warn(
                f"LOCK ON | TTC={ttc_min:.2f}s | vx={self.vx:.2f} | dmin={self.d_min:.2f}"
            )
        elif not self.lock and prev_lock:
            self.get_logger().info("LOCK OFF")

        if self.lock:
            self._stop()

        # Forward block latch
        prev_fb = self.forward_Block

        if self.lock and self.d_min < self.min_distance:
            self.forward_Block = True
        elif self.forward_Block and self.d_min >= self.min_distance:
            self.forward_Block = False

        if self.forward_Block and not prev_fb:
            self.get_logger().warn("FWD_BLOCK ON")
        elif not self.forward_Block and prev_fb:
            self.get_logger().info("FWD_BLOCK OFF")

    # --------------------------------------------------
    def cmd_callback(self, msg):

        self.twist_w = msg.twist.angular.z

        if self.forward_Block:
            if msg.twist.linear.x > 0.0:
                self._stop()
            else:
                self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(msg)

    # --------------------------------------------------
    def _stop(self):
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.twist.linear.x = 0.0
        out.twist.angular.z = self.twist_w
        self.cmd_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AEBNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()