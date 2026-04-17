#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Float32

class AEBNode(Node):

    def __init__(self):
        super().__init__('aeb')

        # ---------- parameters ----------
        self.declare_parameter("ttc_threshold", 0.45)
        self.declare_parameter("secure_min_distance", 0.6)
        self.declare_parameter("robot_radius", 0.45)
        self.declare_parameter("min_speed", 0.05)

        self.ttc_threshold = float(self.get_parameter("ttc_threshold").value)
        self.secure_min_distance = float(self.get_parameter("secure_min_distance").value)
        self.radius = float(self.get_parameter("robot_radius").value)
        self.min_speed = float(self.get_parameter("min_speed").value)

        # ---------- state ----------
        self.lock = False
        self.forward_Block = False

        self.v_ctrl = 0.0
        self.d_min = float('inf')
        self.front_ranges = None
        self.front_angles = None

        self.twist_w = 0.0

        # ---------- subscribers ----------
        self.create_subscription(Float32,   '/lidar/vctrl',      self.vctrl_callback,  10)
        self.create_subscription(Float32,   '/lidar/d_min',      self.dmin_callback,   10)
        self.create_subscription(LaserScan, '/lidar/front_scan', self.scan_callback,   10)

        self.create_subscription(TwistStamped, '/cmd_vel_joy',  self.cmdjoy_callback,  10)
        self.create_subscription(TwistStamped, '/cmd_vel_ctrl', self.cmdctrl_callback, 10)

        # ---------- publishers ----------
        self.cmd_pub  = self.create_publisher(TwistStamped, '/cmd_vel_safe', 10)
        self.dist_pub = self.create_publisher(Twist,        '/dist_min',     10)

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def vctrl_callback(self, msg):
        self.v_ctrl = msg.data

    def dmin_callback(self, msg):
        self.d_min = msg.data

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        self.front_ranges = ranges
        self.front_angles = angles

        self.process_aeb()

    # --------------------------------------------------
    # TTC
    # --------------------------------------------------

    def _ttc_calculus(self):

        vx = self.v_ctrl
        if vx < self.min_speed or self.front_ranges is None:
            return float('inf')

        v_closing = vx * np.cos(self.front_angles)
        mask = v_closing > 0.1

        if not np.any(mask):
            return float('inf')

        dist = self.front_ranges[mask] - self.radius

        if np.any(dist <= 0.0):
            return 0.0

        ttc = dist / v_closing[mask]
        return float(np.min(ttc)) if ttc.size else float('inf')

    # --------------------------------------------------
    # STOP
    # --------------------------------------------------

    def _stop(self):
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.twist.linear.x = 0.0
        out.twist.angular.z = self.twist_w
        self.cmd_pub.publish(out)
        self.twist_w = 0.0

    # --------------------------------------------------
    # Forward block
    # --------------------------------------------------

    def _forward_Block_(self, msg):
        if self.forward_Block:
            if msg.twist.linear.x > 0:
                self._stop()
                self.get_logger().warn("FWD_BLOCK: forward blocked")

    # --------------------------------------------------
    # MAIN AEB LOGIC
    # --------------------------------------------------

    def process_aeb(self):

        ttc_min = self._ttc_calculus()
        prev_lock = self.lock

        # debug
        dist_msg = Twist()
        dist_msg.linear.x = self.d_min
        dist_msg.linear.y = self.v_ctrl
        self.dist_pub.publish(dist_msg)

        # lock condition
        self.lock = (self.d_min <= 1.4) and (ttc_min < self.ttc_threshold)

        if self.lock and not prev_lock:
            self.get_logger().warn(
                f"LOCK ON  | TTC={ttc_min:.2f}s | v_ctrl={self.v_ctrl:.2f} | dmin={self.d_min:.2f}"
            )
        elif not self.lock and prev_lock:
            self.get_logger().info(
                f"LOCK OFF | TTC={ttc_min:.2f}s | v_ctrl={self.v_ctrl:.2f} | dmin={self.d_min:.2f}"
            )

        if self.lock:
            self._stop()

        # forward block
        prev_fb = self.forward_Block

        if self.lock and (self.d_min < self.secure_min_distance):
            self.forward_Block = True
        elif self.forward_Block and (self.d_min >= self.secure_min_distance):
            self.forward_Block = False

        if self.forward_Block and not prev_fb:
            self.get_logger().warn("FWD_BLOCK ON")
        elif not self.forward_Block and prev_fb:
            self.get_logger().info("FWD_BLOCK OFF")

    # --------------------------------------------------
    # CONTROL INPUTS
    # --------------------------------------------------

    def cmdjoy_callback(self, msg):
        self.twist_w = msg.twist.angular.z
        self._forward_Block_(msg)

    def cmdctrl_callback(self, msg):
        self.twist_w = msg.twist.angular.z
        self._forward_Block_(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AEBNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()