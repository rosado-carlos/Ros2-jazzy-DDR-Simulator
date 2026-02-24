#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool


class LidarFeatures(Node):
    """
    Feature-extractor de LiDAR para:
      - distancia frontal robusta (percentiles)
      - estimación de velocidad ego (vx) SOLO con LiDAR (diferenciado por rayo)
      - error/alpha para wall-follow
      - detección de corner con histéresis (frontal cercano + ángulo entre paredes)
    Publica:
      /lidar/front_p10, /lidar/front_median, /lidar/front_min
      /lidar/vx
      /wall/alpha, /wall/error
      /corner
    """

    def __init__(self):
        super().__init__("lidar_features")

        # ---------------- Parameters ----------------
        # Sectors
        self.declare_parameter("front_angle_deg", 20.0)      # ±front
        self.declare_parameter("vx_sector_deg", 15.0)        # ± para estimar vx (estrecho)
        self.declare_parameter("side_center_deg", -90.0)     # derecha=-90, izquierda=+90
        self.declare_parameter("side_width_deg", 20.0)       # ventana lateral total (p.ej. 20 => ±10)

        # vx estimation
        self.declare_parameter("min_speed", 0.05)
        self.declare_parameter("max_speed", 5.0)
        self.declare_parameter("min_cos", 0.3)               # evita dividir por cos pequeño
        self.declare_parameter("vx_lpf_alpha", 0.6)         # 0.7-0.95 (más alto = más suave)

        # Corner detection
        self.declare_parameter("front_enter_threshold", 3.0)
        self.declare_parameter("front_exit_threshold", 3.3)
        self.declare_parameter("corner_angle_threshold_deg", 35.0)  # angulo entre direcciones (PCA)
        self.declare_parameter("pca_min_points", 8)

        # ---------------- Read params ----------------
        self.front_angle = math.radians(float(self.get_parameter("front_angle_deg").value))
        self.vx_sector = math.radians(float(self.get_parameter("vx_sector_deg").value))

        self.side_center = math.radians(float(self.get_parameter("side_center_deg").value))
        self.side_width = math.radians(float(self.get_parameter("side_width_deg").value))


        self.min_speed = float(self.get_parameter("min_speed").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.min_cos = float(self.get_parameter("min_cos").value)
        self.vx_alpha = float(self.get_parameter("vx_lpf_alpha").value)

        self.front_enter = float(self.get_parameter("front_enter_threshold").value)
        self.front_exit = float(self.get_parameter("front_exit_threshold").value)
        self.corner_angle_thr = math.radians(float(self.get_parameter("corner_angle_threshold_deg").value))
        self.pca_min_points = int(self.get_parameter("pca_min_points").value)

        # ---------------- State ----------------
        self.prev_ranges = None
        self.prev_t = None
        self.vx_f = 0.0
        self.corner_active = False

        # ---------------- Pub/Sub ----------------
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 1)

        self.pub_front_p10 = self.create_publisher(Float32, "/lidar/front_p10", 2)
        self.pub_front_med = self.create_publisher(Float32, "/lidar/front_median", 2)
        self.pub_front_min = self.create_publisher(Float32, "/lidar/front_min", 2)
        self.pub_vx = self.create_publisher(Float32, "/lidar/vx", 2)

        self.pub_corner = self.create_publisher(Bool, "/corner", 2)

        self.get_logger().info("lidar_features started")

    # ---------------- Helpers ----------------
    def _angle_to_index(self, scan: LaserScan, a: float) -> int:
        return int(round((a - scan.angle_min) / scan.angle_increment))

    def _sector_vals(self, scan: LaserScan, ranges_np: np.ndarray, a0: float, a1: float):
        i0 = max(0, self._angle_to_index(scan, a0))
        i1 = min(len(ranges_np) - 1, self._angle_to_index(scan, a1))
        if i1 < i0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        seg = ranges_np[i0:i1 + 1]
        idxs = np.arange(i0, i1 + 1)
        thetas = scan.angle_min + idxs * scan.angle_increment

        mask = np.isfinite(seg) & (scan.range_min < seg) & (seg < scan.range_max)
        return seg[mask], thetas[mask]

    def _pstats(self, vals: np.ndarray, range_max: float):
        if vals.size == 0:
            return range_max, range_max, range_max  # p10, median, min
        p10 = float(np.percentile(vals, 10))
        med = float(np.median(vals))
        vmin = float(np.min(vals))
        return p10, med, vmin

    def _pca_dir(self, rs: np.ndarray, thetas: np.ndarray):
        # devuelve vector unitario dirección principal (PCA) en 2D, o None
        if rs.size < self.pca_min_points:
            return None
        xs = rs * np.cos(thetas) 
        ys = rs * np.sin(thetas) 
        P = np.column_stack((xs, ys)) 
        P -= P.mean(axis=0, keepdims=True) 
        U, S, Vt = np.linalg.svd(P, full_matrices=False) 
        v = Vt[0] # dirección principal 
        v = v / np.linalg.norm(v)
        return v

    def _angle_between(self, u: np.ndarray, v: np.ndarray) -> float:
        # ángulo [0, pi]
        dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
        return float(math.acos(dot))

    # ---------------- Main callback ----------------
    def scan_callback(self, scan: LaserScan):
        st = scan.header.stamp
        t = float(st.sec) + 1e-9 * float(st.nanosec)

        ranges = np.asarray(scan.ranges, dtype=np.float64)
        n = len(ranges)

        # ---- Front stats (robust) ----
        r_front, th_front = self._sector_vals(scan, ranges, -self.front_angle, +self.front_angle)
        front_p10, front_med, front_min = self._pstats(r_front, scan.range_max)

        # publish front
        m = Float32(); m.data = front_p10; self.pub_front_p10.publish(m)
        m = Float32(); m.data = front_med; self.pub_front_med.publish(m)
        m = Float32(); m.data = front_min; self.pub_front_min.publish(m)

        # ---- vx from LiDAR (differencing in a narrow frontal sector) ----
        vx_new = 0.0
        if self.prev_ranges is not None and self.prev_t is not None:
            dt = t - self.prev_t
            if 1e-4 < dt < 0.5:
                # indices for vx sector
                i0 = max(0, self._angle_to_index(scan, -self.vx_sector))
                i1 = min(n - 1, self._angle_to_index(scan, +self.vx_sector))
                idxs = np.arange(i0, i1 + 1)
                thetas = scan.angle_min + idxs * scan.angle_increment
                c = np.cos(thetas)

                v_candidates = []
                prev = self.prev_ranges
                for k, i in enumerate(idxs):
                    if abs(c[k]) < self.min_cos:
                        continue
                    r = ranges[i]
                    rp = prev[i] if i < len(prev) else np.inf
                    if not (np.isfinite(r) and np.isfinite(rp)):
                        continue
                    if not (scan.range_min < r < scan.range_max and scan.range_min < rp < scan.range_max):
                        continue
                    vi = (rp - r) / (dt * c[k])
                    if 0.0 <= vi <= self.max_speed:
                        v_candidates.append(vi)

                if v_candidates:
                    vx_new = float(np.median(v_candidates))

        if (not np.isfinite(vx_new)) or vx_new < self.min_speed:
            vx_new = 0.0

        # low-pass filter
        self.vx_f = self.vx_alpha * self.vx_f + (1.0 - self.vx_alpha) * vx_new

        mv = Float32(); mv.data = float(self.vx_f); self.pub_vx.publish(mv)

        # update memory
        self.prev_ranges = ranges.copy()
        self.prev_t = t

        # Lateral sector
        a0_side = self.side_center - 0.5 * self.side_width
        a1_side = self.side_center + 0.5 * self.side_width
        r_side, th_side = self._sector_vals(scan, ranges, a0_side, a1_side)

        # ---- Corner detection robusta: frontal cerca + ángulo entre direcciones (PCA) + histéresis ----
        mask_right = th_front < 0.0

        dir_front = self._pca_dir(r_front[mask_right], th_front[mask_right])
        dir_side = self._pca_dir(r_side, th_side)

        angle_diff = 0.0
        if dir_front is not None and dir_side is not None:
            angle_diff = self._angle_between(dir_front, dir_side)

            
        enter = (front_p10 < self.front_enter) and (abs(angle_diff - math.pi/2) < self.corner_angle_thr)
        #enter = (front_p10 < self.front_enter) and (angle_diff > self.corner_angle_thr)
        exit_ = (front_p10 > self.front_exit) or (abs(angle_diff - math.pi/2) > 2*self.corner_angle_thr)
        #exit_ = (front_p10 > self.front_exit) or (angle_diff < 0.5 * self.corner_angle_thr)

        if (not self.corner_active) and enter:
            self.corner_active = True
        elif self.corner_active and exit_:
            self.corner_active = False

        cb = Bool()
        cb.data = bool(self.corner_active)
        self.pub_corner.publish(cb)


def main(args=None):
    rclpy.init(args=args)
    node = LidarFeatures()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()