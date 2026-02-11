#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped


class TTCFollowGap(Node):

    def __init__(self):
        super().__init__('ttc_follow_gap')

        # parámetros
        self.declare_parameter('ttc_min', 1.0)
        self.declare_parameter('v_max', 0.8)
        self.declare_parameter('kp_steering', 1.5)

        self.ttc_min = self.get_parameter('ttc_min').value
        self.v_max = self.get_parameter('v_max').value
        self.kp = self.get_parameter('kp_steering').value

        self.current_speed = 0.5

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel_nav',
            10
        )

        self.get_logger().info("TTC Follow The Gap activo")

    def scan_callback(self, msg):

        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # limpiar datos
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = 0.0

        v = self.current_speed

        # calcular TTC
        ttc = np.full_like(ranges, np.inf)

        for i in range(len(ranges)):
            if math.cos(angles[i]) > 0.0:
                closing_speed = v * math.cos(angles[i])
                if closing_speed > 0.01:
                    ttc[i] = ranges[i] / closing_speed

        # máscara segura
        safe = ttc > self.ttc_min

        # encontrar gaps
        gaps = []
        start = None

        for i in range(len(safe)):
            if safe[i] and start is None:
                start = i
            elif not safe[i] and start is not None:
                gaps.append((start, i))
                start = None

        if start is not None:
            gaps.append((start, len(safe)-1))

        if not gaps:
            return

        # gap más grande
        largest = max(gaps, key=lambda g: g[1]-g[0])
        center = int((largest[0] + largest[1]) / 2)

        target_angle = angles[center]

        steering = self.kp * target_angle

        # velocidad según TTC mínimo
        min_ttc = np.min(ttc)
        speed = min(self.v_max, min_ttc / 2.0)
        speed = max(0.0, speed)

        self.current_speed = speed

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        cmd.twist.linear.x = speed
        cmd.twist.angular.z = steering

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TTCFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
