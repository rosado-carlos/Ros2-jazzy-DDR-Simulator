#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan

class AEBNode(Node):

    def __init__(self):
        super().__init__('aeb')

        # ---------- thresholds ----------
        self.ttc_threshold = 0.45
        self.secure_min_distance = 0.8
        self.safety_margin = 0.1

        # ---------- geometry ----------
        self.radius = 0.45
        self.eps_closing = 0.1

        # ---------- brake control ----------
        self.brake_active = False
        self.min_speed = 0.05

        # ---------- state ----------
        self.lock = False
        self.forward_Block = False

        self.vx = 0.0
        self.d_min = 0.0
        self.front_scan = None
        self.twist_w = 0.0

        # ---------- subs ----------
        self.vx_sub = self.create_subscription(Float32, '/lidar/vx', self.vx_callback, 10)
        self.dmin_sub = self.create_subscription(Float32, '/lidar/d_min', self.dmin_callback, 10)
        self.front_sub = self.create_subscription(LaserScan, '/lidar/front_scan', self.front_callback, 1)

        self.cmdjoy_sub = self.create_subscription(TwistStamped, '/cmd_vel_joy', self.cmdjoy_callback, 10)
        self.cmdctrl_sub = self.create_subscription(TwistStamped, '/cmd_vel_ctrl', self.cmdctrl_callback, 10)
        self.cmdkey_sub = self.create_subscription(TwistStamped, '/cmd_vel_key', self.cmdkey_callback, 10)

        # ---------- pub ----------
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_safe', 10)

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
    # BRAKE
    # --------------------------------------------------
    def _brake_step(self):
        # Publish brake command
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.twist.linear.x = -3.0  # comando de frenado agresivo
        out.twist.angular.z = 0.0
        self.brake_active = True
        self.cmd_pub.publish(out)

    # --------------------------------------------------
    # TTC
    # --------------------------------------------------
    def _compute_ttc(self):

        if self.front_scan is None:
            return float('inf')

        ranges = np.array(self.front_scan.ranges)

        angles = self.front_scan.angle_min + np.arange(len(ranges)) * self.front_scan.angle_increment

        # filter valid ranges
        mask = np.isfinite(ranges)
        ranges = ranges[mask]
        angles = angles[mask]

        if self.vx <= 0.0 or ranges.size == 0:
            return float('inf')

        v_closing = self.vx * np.cos(angles)
        valid = v_closing > self.eps_closing

        if not np.any(valid):
            return float('inf')

        dist = ranges[valid] - self.radius

        if np.any(dist <= self.safety_margin):
            return 0.0

        ttc = dist / v_closing[valid]
        return float(np.min(ttc)) if ttc.size else float('inf')

    # --------------------------------------------------
    # AEB logic
    # --------------------------------------------------
    def process_aeb(self):

        ttc_min = self._compute_ttc()
        prev_lock = self.lock

        if self.d_min > 1.4:
            self.ttc_threshold = 0.45
        elif self.d_min > 1.0:
            self.ttc_threshold = 0.86
        elif self.d_min > 0.8:
            self.ttc_threshold = 1.8
        else:
            self.ttc_threshold = 3.0

        if not prev_lock and (ttc_min < self.ttc_threshold) and (self.d_min <= 1.8):
            self.lock = True
        elif self.lock and (ttc_min > self.ttc_threshold * 1.2):
            self.lock = False
        
        # ---------- DEBUG LOGS ----------
        if self.lock and not prev_lock:
            self.get_logger().warn(
                "\nLOCK ON\n"
                "-----------------------------\n"
                f"TTC min      : {ttc_min:.3f} s\n"
                f"vx           : {self.vx:.3f} m/s\n"
                f"d_min        : {self.d_min:.3f} m\n"
                f"threshold    : {self.ttc_threshold:.3f} s\n"
                "-----------------------------"
            )

        elif (not self.lock) and prev_lock:
            self.get_logger().info(
                f"LOCK OFF | TTC={ttc_min:.3f}s | vx={self.vx:.2f} | d={self.d_min:.2f}"
            )

        if self.lock and self.vx > 0.1:
            if self.vx > self.min_speed and not self.brake_active:
                self._brake_step()
                self.get_logger().warn("BRAKE START")
        else:
            if self.brake_active and self.vx <= self.min_speed:
                self.brake_active = False
                self._stop()
                self.get_logger().info("BRAKE STOP | lock released")

        if self.lock and self.d_min < self.secure_min_distance:
            self.forward_Block = True

        # ---------- forward block ----------
        if self.forward_Block and self.d_min >= self.secure_min_distance:
            self.forward_Block = False
            self.get_logger().info("FWD_BLOCK OFF | safe distance recovered")

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------
    def vx_callback(self, msg):
        self.vx = msg.data

    def dmin_callback(self, msg):
        self.d_min = msg.data

    def front_callback(self, msg):
        self.front_scan = msg
        self.process_aeb()

    # --------------------------------------------------
    # Command filtering
    # --------------------------------------------------
    def _forward_block_logic(self, msg):
        if self.forward_Block:
            if msg.twist.linear.x > 0:
                self._stop()
                self.get_logger().warn("FWD_BLOCK | forward command rejected")
            else:
                self.get_logger().debug("FWD_BLOCK | reverse allowed")

    def cmdjoy_callback(self, msg):
        self.twist_w = msg.twist.angular.z
        self._forward_block_logic(msg)

    def cmdctrl_callback(self, msg):
        self.twist_w = msg.twist.angular.z
        self._forward_block_logic(msg)

    def cmdkey_callback(self, msg):
        self.twist_w = msg.twist.angular.z
        self._forward_block_logic(msg)


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