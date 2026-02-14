#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped


class AEBNode(Node):

    def __init__(self):
        super().__init__('aeb')

        # ---------- parámeters ----------
        self.declare_parameter('ttc_threshold', 0.5)
        self.declare_parameter('front_angle_deg', 30.0)
        self.declare_parameter('min_speed', 0.05)

        # ---------- geometry ----------
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('eps_closing', 0.1)

        self.declare_parameter('v_sector_deg', 45.0)

        self.declare_parameter('reverse_unblock_speed', 0.05)

        self.declare_parameter('min_distance_for_block', 0.8)

        self.ttc_threshold = float(self.get_parameter('ttc_threshold').value)
        self.front_angle = np.deg2rad(float(self.get_parameter('front_angle_deg').value))
        self.v_sector = np.deg2rad(float(self.get_parameter('v_sector_deg').value))
        self.min_speed = float(self.get_parameter('min_speed').value)

        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.eps_closing = float(self.get_parameter('eps_closing').value)
        self.reverse_unblock_speed = float(self.get_parameter('reverse_unblock_speed').value)
        self.min_distance = float(self.get_parameter('min_distance_for_block').value)

        # ---------- estado ----------
        self.last_scan_time = None
        self.prev_ranges = None
        self.prev_stamp = None
        self.v_est = 0.0
        self.blocking = False
        self.forward_locked = False
        self.min_front_distance = float('inf')

        # ---------- subscripciones ----------
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmdjoy_sub = self.create_subscription(
            TwistStamped, 
            '/cmd_vel_joy', 
            self.cmdjoy_callback, 
            10)


        # ---------- publisher ----------
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel_safe',
            10
        )

        self.get_logger().info(
            f"AEB iniciado | TTC<{self.ttc_threshold}s | frente ±{np.rad2deg(self.front_angle):.0f}°"
        )

    # --------------------------------------------------
    # LIDAR
    # --------------------------------------------------

    def angle_to_index(self, msg: LaserScan, a: float) -> int:
        return int(round((a - msg.angle_min) / msg.angle_increment))

    def valid_range(self, msg: LaserScan, r: float) -> bool:
        return np.isfinite(r) and (msg.range_min < r < msg.range_max)
    
    def cmdjoy_callback(self, msg: TwistStamped):
        vx = float(msg.twist.linear.x)

        # liberar el "forward lock" SOLO si ya hay distancia suficiente
        if self.forward_locked and self.min_front_distance > self.min_distance:
            self.forward_locked = False
            self.get_logger().info(
                f"FORWARD ENABLED | d_front={self.min_front_distance:.2f} > {self.min_distance:.2f}"
            )

        # Si está bloqueado por TTC: permitir reversa para salir, si no -> STOP
        if self.blocking:
            out = TwistStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = msg.header.frame_id

            if vx < -self.reverse_unblock_speed:
                # sale del bloqueo TTC, pero forward_locked puede seguir True
                self.blocking = False
                out.twist.linear.x = vx
                out.twist.angular.z = 0.0
                self.cmd_pub.publish(out)
                self.get_logger().info(
                    "BLOCKED STATE BY TTC RELEASE"
                )
                return

            out.twist.linear.x = 0.0
            out.twist.angular.z = 0.0
            self.cmd_pub.publish(out)
            return

        if self.forward_locked and vx > self.min_speed:

            out = TwistStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = msg.header.frame_id
            out.twist.linear.x = 0.0
            out.twist.angular.z = 0.0
            self.cmd_pub.publish(out)
            return

        # si no está bloqueado y no está forward_locked -> no haces nada (tu mux deja pasar cmd_vel_joy)


    def scan_callback(self, msg):
        should_block = self.blocking

        self.last_scan_time = self.get_clock().now()

        stamp = msg.header.stamp
        stamp_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)

        if self.prev_ranges is None or self.prev_stamp is None:
            self.prev_ranges = list(msg.ranges)
            self.prev_stamp = stamp_s
            return
        
        dt = stamp_s - self.prev_stamp
        if dt <= 1e-4:
            # avoid division by zero / bad stamps
            self.prev_ranges = list(msg.ranges)
            self.prev_stamp = stamp_s
            return
        
        if dt > 0.5:
            self.prev_ranges = list(msg.ranges)
            self.prev_stamp = stamp_s
            return

        i0v = max(0, self.angle_to_index(msg, -self.v_sector))
        i1v = min(len(msg.ranges) - 1, self.angle_to_index(msg, +self.v_sector))

        v_candidates = []
        for i in range(i0v, i1v + 1):
            r = msg.ranges[i]
            r_prev = self.prev_ranges[i] if i < len(self.prev_ranges) else float('inf')

            if not (self.valid_range(msg, r) and self.valid_range(msg, r_prev)):
                continue

            theta = msg.angle_min + i * msg.angle_increment
            c = np.cos(theta)

            if abs(c) < 0.3:
                continue
                
            v_i = (r_prev - r) / (dt * c)

            if v_i < 0.0 or v_i > 5.0:
                continue
        
            v_candidates.append(v_i)
        
        if len(v_candidates) == 0:
            v_raw = 0.0
        else:
            v_raw = float(np.median(v_candidates))
        if not np.isfinite(v_raw):
            v_raw = 0.0

        self.v_est = v_raw
            
        if abs(self.v_est) < self.min_speed:
            self.v_est = 0.0
            
        i0 = max(0, self.angle_to_index(msg, -self.front_angle))
        i1 = min(len(msg.ranges) - 1, self.angle_to_index(msg, +self.front_angle))

        min_d = float('inf')
        for rr in msg.ranges[i0:i1 + 1]:
            if self.valid_range(msg, rr) and rr < min_d:
                min_d = rr
        self.min_front_distance = min_d

        ttc_min = float('inf')

        if self.v_est > 0.0:
            for i in range(i0, i1 + 1):
                r = msg.ranges[i]
                if not self.valid_range(msg, r):
                    continue

                # already inside inflated radius
                if r <= self.robot_radius:
                    ttc_min = 0.0
                    break

                theta = msg.angle_min + i * msg.angle_increment
                v_closing = self.v_est * np.cos(theta)
                if v_closing <= self.eps_closing:
                    continue

                ttc_i = (r - self.robot_radius) / v_closing
                if ttc_i < ttc_min:
                    ttc_min = ttc_i


        if not self.blocking:
            should_block = (ttc_min < self.ttc_threshold)

        if should_block != self.blocking:
            self.blocking = should_block
            if should_block:
                self.forward_locked = True
                self.get_logger().warn(
                    f"STOP | v={self.v_est:.2f} m/s TTCmin={ttc_min:.2f} s robot_radius={self.robot_radius:.2f}m"
                )
            else:
                self.get_logger().info("CLEAR | movimiento permitido")

        # publish command
        out = TwistStamped()
        out.header.stamp = self.last_scan_time.to_msg()
        out.header.frame_id = msg.header.frame_id

        out.twist.linear.x = 0.0
        out.twist.angular.z = 0.0

        if should_block:
            out.twist.linear.x = 0.0
            out.twist.angular.z = 0.0
            self.cmd_pub.publish(out)
            
        self.prev_ranges = list(msg.ranges)
        self.prev_stamp = stamp_s


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