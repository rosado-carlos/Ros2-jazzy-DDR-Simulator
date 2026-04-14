#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

class LidarDataNode(Node):

    def __init__(self):
        super().__init__('lidar_data')

        # ---------- configuration ----------
        self.front_angle = np.deg2rad(20.0)
        self.v_sector = np.deg2rad(30.0)
        self.min_speed = 0.05
        self.max_speed = 5.0

        # ---------- state ----------
        self.prev_ranges = None
        self.prev_stamp = None

        self.d_min = 0.0
        self.vx_filt = 0.0

        # ---------- ROS interfaces ----------
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)

        self.vx_pub = self.create_publisher(Float32, '/lidar/vx', 10)
        self.dmin_pub = self.create_publisher(Float32, '/lidar/d_min', 10)
        self.front_pub = self.create_publisher(LaserScan, '/lidar/front_scan', 10)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _angle_to_index(self, scan_msg, a):
        return int(round((a - scan_msg.angle_min) / scan_msg.angle_increment))

    def _valid_range(self, scan_msg, r):
        return np.isfinite(r) and (scan_msg.range_min < r < scan_msg.range_max)

    # --------------------------------------------------
    # Sector extraction (CONSISTENT CONTRACT)
    # --------------------------------------------------
    def _sector_ranges(self, lidar_msg, ranges, a0, a1):

        i0 = max(0, self._angle_to_index(lidar_msg, a0))
        i1 = min(len(ranges) - 1, self._angle_to_index(lidar_msg, a1))

        if i1 < i0:
            return None, None, None, None

        seg = ranges[i0:i1+1]

        mask = np.isfinite(seg) & (lidar_msg.range_min < seg) & (seg < lidar_msg.range_max)

        return seg, mask, i0, i1

    # --------------------------------------------------
    # Kinematics
    # --------------------------------------------------
    def _kin_state_by_lidar(self, lidar_msg: LaserScan):

        st = lidar_msg.header.stamp
        t = float(st.sec) + 1e-9 * float(st.nanosec)

        ranges = np.asarray(lidar_msg.ranges, dtype=np.float64)

        seg, mask, i0, i1 = self._sector_ranges(
            lidar_msg, ranges, -self.front_angle, +self.front_angle
        )

        if seg is None:
            return 0.0, (None, None, None, None), float('inf')

        valid_ranges = seg[mask]
        self.d_min = float(np.min(valid_ranges)) if valid_ranges.size else float('inf')

        # -------- FIRST FRAME --------
        if self.prev_ranges is None or self.prev_stamp is None:
            self.prev_ranges = ranges.copy()
            self.prev_stamp = t
            return 0.0, (seg, mask, i0, i1), self.d_min

        dt = t - self.prev_stamp

        if dt <= 1e-4 or dt > 0.5:
            self.get_logger().warn(f"[LIDAR] dt inválido: {dt:.6f}s")
            self.prev_ranges = ranges.copy()
            self.prev_stamp = t
            return 0.0, (seg, mask, i0, i1), self.d_min

        prev = self.prev_ranges

        # -------- VELOCITY --------
        i0v = max(0, self._angle_to_index(lidar_msg, -self.v_sector))
        i1v = min(len(ranges) - 1, self._angle_to_index(lidar_msg, +self.v_sector))

        idxs_v = np.arange(i0v, i1v + 1)
        thetas_v = lidar_msg.angle_min + idxs_v * lidar_msg.angle_increment

        vr_list = []
        cos_list = []

        for k, i in enumerate(idxs_v):
            r = ranges[i]
            rp = prev[i] if i < len(prev) else np.inf

            if not (self._valid_range(lidar_msg, r) and self._valid_range(lidar_msg, rp)):
                continue

            ctheta = np.cos(thetas_v[k])

            if abs(ctheta) < 0.5:
                continue

            if abs(r - rp) > 1.5:
                continue

            vr = (rp - r) / dt

            if abs(vr) > self.max_speed:
                continue

            vr_list.append(vr)
            cos_list.append(ctheta)

        if len(vr_list) < 5:
            vx = 0.0
            self.get_logger().warn("[LIDAR] Pocos puntos para estimar velocidad")
        else:
            A = np.array(cos_list).reshape(-1, 1)
            b = np.array(vr_list)

            vx_ls = float(np.linalg.lstsq(A, b, rcond=None)[0][0])
            vx_median = float(np.median(b / A.flatten()))

            vx = 0.7 * vx_ls + 0.3 * vx_median

        if (not np.isfinite(vx)) or abs(vx) < self.min_speed:
            vx = 0.0

        self.prev_ranges = ranges.copy()
        self.prev_stamp = t

        return vx, (seg, mask, i0, i1), self.d_min

    # --------------------------------------------------
    # Callback
    # --------------------------------------------------
    def scan_callback(self, msg):

        vx, dx, self.d_min = self._kin_state_by_lidar(msg)

        # -------- SAFE UNPACK --------
        if dx[0] is None:
            return

        seg, mask, i0, i1 = dx

        # -------- FILTER VX --------
        alpha = 0.5 if abs(vx - self.vx_filt) > 0.5 else 0.2
        self.vx_filt = (1 - alpha) * self.vx_filt + alpha * vx
        vx = self.vx_filt
        if abs(vx) < self.min_speed:
            vx = 0.0

        # -------- DEBUG --------
        if self.d_min < 1.0:
            self.get_logger().warn(f"[LIDAR] Objeto cercano | d_min={self.d_min:.2f} m")

        if np.sum(mask) < 5:
            self.get_logger().warn("[LIDAR] Muy pocos puntos válidos en el frente")

        # -------- PUBLISH SCALARS --------
        vx_msg = Float32()
        vx_msg.data = float(vx)
        self.vx_pub.publish(vx_msg)

        dmin_msg = Float32()
        dmin_msg.data = float(self.d_min)
        self.dmin_pub.publish(dmin_msg)

        # -------- FRONT SCAN --------
        front_scan = LaserScan()
        front_scan.header = msg.header

        filtered_ranges = np.where(mask, seg, np.inf)

        front_scan.angle_min = msg.angle_min + i0 * msg.angle_increment
        front_scan.angle_max = msg.angle_min + i1 * msg.angle_increment
        front_scan.angle_increment = msg.angle_increment

        front_scan.time_increment = msg.time_increment
        front_scan.scan_time = msg.scan_time

        front_scan.range_min = msg.range_min
        front_scan.range_max = msg.range_max

        front_scan.ranges = filtered_ranges.tolist()

        self.front_pub.publish(front_scan)

def main(args=None):
    rclpy.init(args=args)
    node = LidarDataNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()