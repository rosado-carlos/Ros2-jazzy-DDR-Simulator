#!/usr/bin/env python3
"""
speed_estimator_node.py  --  Online speed estimator using fitted 1st-order model.

Subscribes:  /esc/command  (std_msgs/Float32)   normalized ESC command u
Publishes:   /estimated_speed  (std_msgs/Float32)   estimated forward speed [m/s]
             /estimated_odom   (nav_msgs/Odometry)   dead-reckoning odometry
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class SpeedEstimatorNode(Node):

    def __init__(self):
        super().__init__("speed_estimator_node")

        # ── Fitted model parameters (update after running fit_model.py) ──────
        self.declare_parameter("K",      1.0)    # m/s per unit command
        self.declare_parameter("tau",    0.3)    # time constant [s]
        self.declare_parameter("u_dead", 0.05)   # dead zone threshold
        self.declare_parameter("dt",     0.1)    # estimator update rate [s]

        self._K      = self.get_parameter("K").value
        self._tau    = self.get_parameter("tau").value
        self._u_dead = self.get_parameter("u_dead").value
        self._dt     = self.get_parameter("dt").value

        # State
        self._v   = 0.0   # estimated speed [m/s]
        self._x   = 0.0
        self._y   = 0.0
        self._yaw = 0.0
        self._u   = 0.0   # latest ESC command

        # I/O
        self._sub = self.create_subscription(
            Float32, "/esc/command", self._cmd_cb, 10)
        self._pub_speed = self.create_publisher(Float32, "/estimated_speed", 10)
        self._pub_odom  = self.create_publisher(Odometry, "/estimated_odom", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._timer = self.create_timer(self._dt, self._update)
        self.get_logger().info(
            f"SpeedEstimator: K={self._K}, tau={self._tau}, u_dead={self._u_dead}"
        )

    def _cmd_cb(self, msg: Float32):
        self._u = float(msg.data)

    def _apply_deadzone(self, u):
        if u > self._u_dead:
            return u - self._u_dead
        if u < -self._u_dead:
            return u + self._u_dead
        return 0.0

    def _update(self):
        u_eff = self._apply_deadzone(self._u)
        dvdt  = (self._K * u_eff - self._v) / self._tau
        self._v += dvdt * self._dt

        # Dead-reckoning (assumes /cmd_vel also contains yaw rate for steering)
        self._x   += self._v * math.cos(self._yaw) * self._dt
        self._y   += self._v * math.sin(self._yaw) * self._dt

        # Publish speed
        self._pub_speed.publish(Float32(data=float(self._v)))

        # Publish odometry
        now = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp    = now
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.twist.twist.linear.x = self._v
        self._pub_odom.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = SpeedEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()