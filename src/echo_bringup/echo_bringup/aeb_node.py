#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped


class AEBNode(Node):

    def __init__(self):
        super().__init__('aeb')

        # ---------- parámetros ----------
        self.declare_parameter('ttc_threshold', 1.0)
        self.declare_parameter('front_angle_deg', 30.0)
        self.declare_parameter('min_speed', 0.05)
        self.declare_parameter('scan_timeout', 0.5)

        self.ttc_threshold = float(self.get_parameter('ttc_threshold').value)
        self.front_angle = math.radians(
            float(self.get_parameter('front_angle_deg').value)
        )
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.scan_timeout = float(self.get_parameter('scan_timeout').value)

        # ---------- estado ----------
        self.min_front_distance = float('inf')
        self.last_scan_time = None
        self.blocking = False

        # ---------- subscripciones ----------
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_sub = self.create_subscription(
            TwistStamped,
            '/cmd_vel_joy',
            self.cmd_callback,
            10
        )

        # ---------- publisher ----------
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel_safe',
            10
        )

        self.get_logger().info(
            f"AEB iniciado | TTC<{self.ttc_threshold}s | frente ±{math.degrees(self.front_angle):.0f}°"
        )

    # --------------------------------------------------
    # LIDAR
    # --------------------------------------------------
    def scan_callback(self, msg):

        self.last_scan_time = self.get_clock().now()

        def angle_to_index(a):
            return int(round((a - msg.angle_min) / msg.angle_increment))

        i0 = max(0, angle_to_index(-self.front_angle))
        i1 = min(len(msg.ranges) - 1, angle_to_index(+self.front_angle))

        min_d = float('inf')

        for r in msg.ranges[i0:i1 + 1]:
            if msg.range_min < r < msg.range_max and math.isfinite(r):
                if r < min_d:
                    min_d = r

        self.min_front_distance = min_d

    # --------------------------------------------------
    # CMD VEL
    # --------------------------------------------------
    def cmd_callback(self, msg):

        now = self.get_clock().now()

        scan_stale = (
            self.last_scan_time is None or
            (now - self.last_scan_time).nanoseconds * 1e-9 > self.scan_timeout
        )

        v = float(msg.twist.linear.x)

        should_block = False
        ttc = float('inf')

        if not scan_stale and v > self.min_speed and math.isfinite(self.min_front_distance):
            ttc = self.min_front_distance / v
            should_block = ttc < self.ttc_threshold

        if should_block != self.blocking:
            self.blocking = should_block
            if should_block:
                self.get_logger().warn(
                    f"STOP | d={self.min_front_distance:.2f}m v={v:.2f}m/s TTC={ttc:.2f}s"
                )
            else:
                self.get_logger().info("CLEAR | movimiento permitido")

        out = TwistStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = msg.header.frame_id
        out.twist = msg.twist

        if should_block:
            out.twist.linear.x = 0.0
            out.twist.angular.z = 0.0

        self.cmd_pub.publish(out)


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