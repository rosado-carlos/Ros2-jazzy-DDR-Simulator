#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class FollowGapFinder(Node):
    def __init__(self):
        super().__init__('ttc_gap_finder')

        # -------- PARÁMETROS ROS (leídos desde launch) --------
        self.declare_parameter('ttc_min',       0.6)
        self.declare_parameter('fov_deg',      120.0)
        self.declare_parameter('bubble_base',   0.35)
        self.declare_parameter('bubble_vel_k',  0.4)
        self.declare_parameter('min_clearance', 0.40)  # distancia mínima a pared (m)
        self.declare_parameter('smooth_alpha',  0.40)  # ↓ más responsivo (era 0.7)
        self.declare_parameter('min_depth_threshold', 1.2)  # <1.2m = probable callejón
        self.declare_parameter('deadend_weight',      1.8)  # fuerza del rechazo
        self.min_depth_threshold = self.get_parameter('min_depth_threshold').value
        self.deadend_weight      = self.get_parameter('deadend_weight').value

        self.ttc_threshold = self.get_parameter('ttc_min').value
        self.fov           = math.radians(self.get_parameter('fov_deg').value)
        self.bubble_base   = self.get_parameter('bubble_base').value
        self.bubble_vel_k  = self.get_parameter('bubble_vel_k').value
        self.min_clearance = self.get_parameter('min_clearance').value
        self.alpha         = self.get_parameter('smooth_alpha').value
        self.min_range     = 0.05
        # ... tus declares existentes ...
        self.no_gap_counter = 0
        self.search_dir     = 1.0  # 1.0 o -1.0

        # -------- SUBS --------
        self.create_subscription(LaserScan, '/scan',     self.scan_callback, 10)
        self.create_subscription(Float32,  '/lidar/vx', self.vx_callback,   10)

        # -------- PUBS --------
        self.pub_angle = self.create_publisher(Float32, '/gap_angle', 10)
        self.pub_ttc   = self.create_publisher(Float32, '/min_ttc',   10)

        # -------- ESTADO --------
        self.vx         = 0.0
        self.prev_angle = 0.0

        self.get_logger().info(
            f"GapFinder OK | ttc_min={self.ttc_threshold} "
            f"clearance={self.min_clearance} alpha={self.alpha}"
        )

    # -------------------------------------------------------
    def vx_callback(self, msg: Float32):
        self.vx = msg.data

    # -------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        vx = max(self.vx, 0.05)

        ranges = np.array(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # -------- LIMPIEZA --------
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, self.min_range, msg.range_max)

        # -------- FOV --------
        fov_mask = np.abs(angles) <= self.fov
        ranges   = np.where(fov_mask, ranges, 0.0)

        # -------- TTC  (ANTES del bubble, solo rangos válidos) --------
        safe_r      = np.where(ranges > self.min_range, ranges, np.inf)
        projections = vx * np.cos(angles)
        ttc         = np.where(projections > 0.0, safe_r / projections, np.inf)
        min_ttc     = float(np.min(ttc))

        # -------- BUBBLE MÚLTIPLE ----------------------------------------
        # Itera sobre TODOS los puntos dentro del radio de clearance,
        # no solo el más cercano → elimina secciones enteras de pared.
        bubble_radius = self.bubble_base + self.bubble_vel_k * vx
        close_mask    = (ranges > self.min_range) & (ranges < bubble_radius)

        for idx in np.where(close_mask)[0]:
            dist   = ranges[idx]
            b_ang  = math.atan2(bubble_radius, max(dist, 1e-3))
            b_half = int(b_ang / msg.angle_increment)
            b_s    = max(0,           idx - b_half)
            b_e    = min(len(ranges), idx + b_half)
            ranges[b_s:b_e] = 0.0

        # -------- SAFE MASK -----------------------------------------------
        # TTC cubre amenazas frontales.
        # min_clearance cubre paredes laterales (cos≈0 → TTC=∞ pero siguen peligrosas).
        forward_w = np.cos(angles)
        ttc_dyn   = self.ttc_threshold * (0.5 + 0.5 * forward_w)
        safe_mask = (
            (ttc    > ttc_dyn)           &  # seguridad temporal
            (ranges > self.min_clearance)   # seguridad espacial
        )

        # -------- GAPS --------
        gaps = self._find_gaps(safe_mask)

        if not gaps:
            self.no_gap_counter += 1
            # Alterna dirección cada ~0.5s (10 ciclos a 20Hz)
            if self.no_gap_counter > 20:
                self.search_dir *= -1.0
                self.no_gap_counter = 0
            best_angle = 1.5 * self.search_dir  # ~86° alternados
            self.get_logger().warn(f"Sin gaps → búsqueda ({best_angle:+.2f} rad)")
        else:
            best_angle = self._score_gaps(gaps, ranges, angles, msg.range_max)

        # -------- SUAVIZADO --------
        best_angle      = self.alpha * self.prev_angle + (1.0 - self.alpha) * best_angle
        self.prev_angle = best_angle

        # -------- PUBLICAR --------
        a = Float32(); a.data = best_angle; self.pub_angle.publish(a)
        t = Float32(); t.data = min_ttc;    self.pub_ttc.publish(t)

    # -------------------------------------------------------
    def _find_gaps(self, mask):
        gaps, start = [], None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                gaps.append((start, i - 1)); start = None
        if start is not None:
            gaps.append((start, len(mask) - 1))
        return gaps
    
    def _score_gaps(self, gaps, ranges, angles, range_max):
        scores = []
        for gs, ge in gaps:
            width_idx = ge - gs
            gap_ranges = ranges[gs:ge+1]
            
            # 1. Profundidad mínima (detecta si el hueco se cierra rápido)
            min_depth = float(np.min(gap_ranges))
            
            # 2. Profundidad media (cuánto espacio real hay dentro)
            avg_depth = float(np.mean(gap_ranges))
            
            # 3. Ángulo central
            center_angle = float(angles[gs + width_idx // 2])
            
            # 4. Penalización si es un callejón (poco profundo o se cierra bruscamente)
            deadend_pen = 0.0
            if min_depth < self.min_depth_threshold:
                deadend_pen = self.deadend_weight * (1.0 - min_depth / self.min_depth_threshold)
            
            # 5. Puntuación normalizada
            w_norm = width_idx / len(angles)
            d_norm = avg_depth / range_max
            score = (0.4 * w_norm) + (0.4 * d_norm) - deadend_pen
            
            scores.append((score, center_angle))
        
        # Devuelve el ángulo del gap con mayor puntuación
        _, best = max(scores, key=lambda x: x[0])
        return best


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(FollowGapFinder())
    rclpy.shutdown()

if __name__ == '__main__':
    main()