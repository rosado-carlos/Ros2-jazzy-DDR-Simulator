#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException


# --------------------------------------------------
# Small helpers
# --------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    #This clamp a scalar into [lo, hi].
    return max(lo, min(hi, x))


def wrap_to_pi(a: float) -> float:
    #This wrap an angle to [-pi, pi].
    return math.atan2(math.sin(a), math.cos(a))


class TEBStaticNode(Node):
    #This class implements a minimal TEB-like local planner for static obstacles.
    #This is an educational version: it optimizes a local band of poses + times.

    def __init__(self):
        super().__init__('teb_controller_node')

        # --------------------------------------------------
        # Parameters: topics and frames
        # --------------------------------------------------
        self.declare_parameter('path_topic', '/bitstar_path')
        self.declare_parameter('odom_topic', '/ekf/odometry')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_teb')
        self.declare_parameter('teb_path_topic', '/teb_local_path')
        self.declare_parameter('base_frame', 'base_link')

        # --------------------------------------------------
        # Parameters: control loop and goal behavior
        # --------------------------------------------------
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('goal_tolerance', 0.20)
        self.declare_parameter('yaw_goal_tolerance', 0.20)
        self.declare_parameter('goal_yaw_kp', 0.1)
        self.declare_parameter('tf_timeout_sec', 0.15)
        self.declare_parameter('eps', 1e-6)

        # --------------------------------------------------
        # Parameters: robot limits
        # --------------------------------------------------
        self.declare_parameter('v_nominal', 0.60)
        self.declare_parameter('max_vel_x', 3.00)
        self.declare_parameter('max_vel_theta', 0.40)
        self.declare_parameter('acc_lim_x', 0.80)
        self.declare_parameter('acc_lim_theta', 2.50)
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('robot_radius', 0.25)
        self.declare_parameter('max_lateral_accel', 1.50)

        # --------------------------------------------------
        # Parameters: local horizon and band discretization
        # --------------------------------------------------
        self.declare_parameter('horizon_length', 3.0)
        self.declare_parameter('resample_ds', 0.15)
        self.declare_parameter('max_samples', 25)
        self.declare_parameter('behind_margin', 0.10)
        self.declare_parameter('min_dt', 0.05)
        self.declare_parameter('max_dt', 0.60)

        # --------------------------------------------------
        # Parameters: obstacle handling (static only)
        # --------------------------------------------------
        self.declare_parameter('min_obstacle_dist', 0.40)
        self.declare_parameter('obstacle_influence_dist', 0.90)
        self.declare_parameter('obstacle_subsample', 4)
        self.declare_parameter('max_obstacle_points', 180)

        # --------------------------------------------------
        # Parameters: optimization weights and step sizes
        # --------------------------------------------------
        self.declare_parameter('max_iterations', 50)
        self.declare_parameter('alpha_xy', 0.10)
        self.declare_parameter('alpha_theta', 0.18)
        self.declare_parameter('alpha_dt', 0.25)
        self.declare_parameter('w_path', 1.20)
        self.declare_parameter('w_smooth', 0.60)
        self.declare_parameter('w_obstacle', 0.50)
        self.declare_parameter('w_kinematics_nh', 0.90)
        self.declare_parameter('w_time', 0.35)
        self.declare_parameter('w_dt_smooth', 0.20)
        self.declare_parameter('w_theta_ref', 0.25)

        # --------------------------------------------------
        # Read parameters
        # --------------------------------------------------
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.teb_path_topic = str(self.get_parameter('teb_path_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.yaw_goal_tolerance = float(self.get_parameter('yaw_goal_tolerance').value)
        self.goal_yaw_kp = float(self.get_parameter('goal_yaw_kp').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.eps = float(self.get_parameter('eps').value)

        self.v_nominal = float(self.get_parameter('v_nominal').value)
        self.max_vel_x = float(self.get_parameter('max_vel_x').value)
        self.max_vel_theta = float(self.get_parameter('max_vel_theta').value)
        self.acc_lim_x = float(self.get_parameter('acc_lim_x').value)
        self.acc_lim_theta = float(self.get_parameter('acc_lim_theta').value)
        self.allow_reverse = bool(self.get_parameter('allow_reverse').value)
        self.robot_radius = float(self.get_parameter('robot_radius').value)
        self.max_lateral_accel = float(self.get_parameter('max_lateral_accel').value)

        self.horizon_length = float(self.get_parameter('horizon_length').value)
        self.resample_ds = float(self.get_parameter('resample_ds').value)
        self.max_samples = int(self.get_parameter('max_samples').value)
        self.behind_margin = float(self.get_parameter('behind_margin').value)
        self.min_dt = float(self.get_parameter('min_dt').value)
        self.max_dt = float(self.get_parameter('max_dt').value)

        self.min_obstacle_dist = float(self.get_parameter('min_obstacle_dist').value)
        self.obstacle_influence_dist = float(self.get_parameter('obstacle_influence_dist').value)
        self.obstacle_subsample = max(1, int(self.get_parameter('obstacle_subsample').value))
        self.max_obstacle_points = int(self.get_parameter('max_obstacle_points').value)

        self.max_iterations = int(self.get_parameter('max_iterations').value)
        self.alpha_xy = float(self.get_parameter('alpha_xy').value)
        self.alpha_theta = float(self.get_parameter('alpha_theta').value)
        self.alpha_dt = float(self.get_parameter('alpha_dt').value)
        self.w_path = float(self.get_parameter('w_path').value)
        self.w_smooth = float(self.get_parameter('w_smooth').value)
        self.w_obstacle = float(self.get_parameter('w_obstacle').value)
        self.w_kinematics_nh = float(self.get_parameter('w_kinematics_nh').value)
        self.w_time = float(self.get_parameter('w_time').value)
        self.w_dt_smooth = float(self.get_parameter('w_dt_smooth').value)
        self.w_theta_ref = float(self.get_parameter('w_theta_ref').value)

        # --------------------------------------------------
        # State variables
        # --------------------------------------------------
        self.path = []  #This stores the latest global path poses.
        self.path_frame = None  #This stores the path frame_id.
        self.has_path = False  #This indicates whether a valid path is available.

        self.current_v = 0.0  #This stores current linear velocity from odom.
        self.current_w = 0.0  #This stores current angular velocity from odom.

        self.obstacles_base = np.empty((0, 2), dtype=np.float64)  #This stores scan obstacle points transformed to base frame.
        self.last_band = None  #This stores the latest optimized band for debug/inspection.
        self.goal_reached = False  #This indicates whether the goal was reached.

        # --------------------------------------------------
        # ROS I/O
        # --------------------------------------------------
        self.path_sub = self.create_subscription(Path, self.path_topic, self.on_path, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.teb_path_pub = self.create_publisher(Path, self.teb_path_topic, 10)

        # --------------------------------------------------
        # TF setup
        # --------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------
        self.timer = self.create_timer(1.0 / max(self.control_rate_hz, 1.0), self.on_timer)

        self.get_logger().info('TEB minimal static node initialized.')

    # --------------------------------------------------
    # Message callbacks
    # --------------------------------------------------
    def on_path(self, msg: Path):
        #This stores the new path and resets goal state.
        self.path = list(msg.poses)
        self.path_frame = msg.header.frame_id if msg.header.frame_id else None
        self.has_path = (len(self.path) > 0) and (self.path_frame is not None)
        self.goal_reached = False

        if self.has_path:
            self.get_logger().info(f'Path received | poses={len(self.path)} | frame={self.path_frame}')
        else:
            self.get_logger().warn('Received empty path or missing frame_id.')

    def on_odom(self, msg: Odometry):
        #This stores current robot twist for acceleration limiting.
        self.current_v = float(msg.twist.twist.linear.x)
        self.current_w = float(msg.twist.twist.angular.z)

    def on_scan(self, msg: LaserScan):
        #This converts the scan into a sparse obstacle cloud in base frame.
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        n = len(ranges)

        if n == 0:
            self.obstacles_base = np.empty((0, 2), dtype=np.float64)
            return

        idxs = np.arange(0, n, self.obstacle_subsample, dtype=np.int32)
        angles = msg.angle_min + idxs * msg.angle_increment
        rr = ranges[idxs]

        mask = np.isfinite(rr) & (msg.range_min < rr) & (rr < msg.range_max)
        if not np.any(mask):
            self.obstacles_base = np.empty((0, 2), dtype=np.float64)
            return

        rr = rr[mask]
        angles = angles[mask]

        pts_scan = np.column_stack((rr * np.cos(angles), rr * np.sin(angles)))
        pts_base = self._transform_points_to_base(pts_scan, msg.header.frame_id)

        if pts_base is None or pts_base.size == 0:
            self.obstacles_base = np.empty((0, 2), dtype=np.float64)
            return

        if len(pts_base) > self.max_obstacle_points:
            self.obstacles_base = pts_base[:self.max_obstacle_points, :]
        else:
            self.obstacles_base = pts_base

    # --------------------------------------------------
    # TF helpers
    # --------------------------------------------------
    def _quat_to_yaw(self, q) -> float:
        #This extracts yaw from a quaternion.
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _lookup_tf(self, target_frame: str, source_frame: str):
        #This gets the latest transform source_frame -> target_frame.
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time(), timeout=Duration(seconds=self.tf_timeout_sec))
            return tf
        except Exception as ex:
            self.get_logger().debug(f'TF lookup failed {source_frame}->{target_frame}: {ex}')
            return None

    def _transform_xytheta_to_target(self, x: float, y: float, yaw: float, tf_msg):
        #This applies a 2D rigid transform to a pose.
        tx = float(tf_msg.transform.translation.x)
        ty = float(tf_msg.transform.translation.y)
        tyaw = self._quat_to_yaw(tf_msg.transform.rotation)
        c = math.cos(tyaw)
        s = math.sin(tyaw)

        xo = tx + c * x - s * y
        yo = ty + s * x + c * y
        yoaw = wrap_to_pi(tyaw + yaw)
        return xo, yo, yoaw

    def _transform_pose_to_base(self, pose_msg: PoseStamped):
        #This transforms a PoseStamped from its frame into base_frame using planar TF.
        if not pose_msg.header.frame_id:
            return None

        tf = self._lookup_tf(self.base_frame, pose_msg.header.frame_id)
        if tf is None:
            return None

        px = float(pose_msg.pose.position.x)
        py = float(pose_msg.pose.position.y)
        pyaw = self._quat_to_yaw(pose_msg.pose.orientation)
        return self._transform_xytheta_to_target(px, py, pyaw, tf)

    def _transform_points_to_base(self, pts: np.ndarray, source_frame: str):
        #This transforms an array of 2D points from source_frame to base_frame.
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        if source_frame == self.base_frame or source_frame == '':
            return pts.copy()

        tf = self._lookup_tf(self.base_frame, source_frame)
        if tf is None:
            return None

        tx = float(tf.transform.translation.x)
        ty = float(tf.transform.translation.y)
        tyaw = self._quat_to_yaw(tf.transform.rotation)
        c = math.cos(tyaw)
        s = math.sin(tyaw)

        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        out = (pts @ rot.T)
        out[:, 0] += tx
        out[:, 1] += ty
        return out

    # --------------------------------------------------
    # Command publishers
    # --------------------------------------------------
    def _publish_cmd(self, vx: float, wz: float):
        #This publishes a TwistStamped command.
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.base_frame
        out.twist.linear.x = float(vx)
        out.twist.angular.z = float(wz)
        self.cmd_pub.publish(out)

    def _stop(self):
        #This publishes a complete stop.
        self._publish_cmd(0.0, 0.0)

    def _publish_band_path(self, band):
        #This publishes the optimized band as a Path in base_frame for RViz.
        if band is None:
            return

        x = band['x']
        y = band['y']
        th = band['th']
        n = len(x)

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame

        for i in range(n):
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x[i])
            p.pose.position.y = float(y[i])
            p.pose.position.z = 0.0
            qz = math.sin(th[i] * 0.5)
            qw = math.cos(th[i] * 0.5)
            p.pose.orientation.z = qz
            p.pose.orientation.w = qw
            msg.poses.append(p)

        self.teb_path_pub.publish(msg)

    # --------------------------------------------------
    # Path preprocessing
    # --------------------------------------------------
    def _deduplicate_points(self, pts: np.ndarray):
        #This removes consecutive duplicates or near-duplicates from a 2D polyline.
        if len(pts) <= 1:
            return pts

        out = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - out[-1]) > 1e-3:
                out.append(pts[i])
        return np.asarray(out, dtype=np.float64)

    def _resample_polyline(self, pts: np.ndarray, ds: float, max_samples: int):
        #This resamples a 2D polyline with approximately uniform spacing.
        if len(pts) < 2:
            return pts

        pts = self._deduplicate_points(pts)
        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)

        if np.sum(seg_len) < self.eps:
            return pts[:1]

        s = np.concatenate(([0.0], np.cumsum(seg_len)))
        total = float(s[-1])

        n_samples = int(math.floor(total / max(ds, self.eps))) + 1
        n_samples = clamp(n_samples, 2, max_samples)
        s_query = np.linspace(0.0, total, int(n_samples))

        xq = np.interp(s_query, s, pts[:, 0])
        yq = np.interp(s_query, s, pts[:, 1])
        return np.column_stack((xq, yq))

    def _compute_reference_theta(self, pts: np.ndarray):
        #This computes heading angles along a resampled polyline.
        if len(pts) == 0:
            return np.array([], dtype=np.float64)
        if len(pts) == 1:
            return np.array([0.0], dtype=np.float64)

        th = np.zeros(len(pts), dtype=np.float64)
        for i in range(len(pts) - 1):
            dp = pts[i + 1] - pts[i]
            th[i] = math.atan2(dp[1], dp[0])
        th[-1] = th[-2]
        return th

    def _build_local_reference(self):
        #This transforms the global path into base_frame, crops a local horizon, and resamples it.
        if not self.has_path:
            return None, None

        pts_local = []
        goal_local = None

        for i, pose in enumerate(self.path):
            pose_b = self._transform_pose_to_base(pose)
            if pose_b is None:
                return None, None

            bx, by, byaw = pose_b
            d = math.hypot(bx, by)

            if i == (len(self.path) - 1):
                goal_local = (bx, by, byaw)

            if bx < -self.behind_margin:
                continue

            if d <= self.horizon_length:
                pts_local.append([bx, by])

        if len(pts_local) == 0:
            if goal_local is None:
                return None, None
            pts_local = [[goal_local[0], goal_local[1]]]

        pts_local = np.asarray(pts_local, dtype=np.float64)

        #This prepends the robot origin because the local trajectory starts at the robot.
        pts_local = np.vstack((np.array([[0.0, 0.0]], dtype=np.float64), pts_local))
        pts_local = self._deduplicate_points(pts_local)
        pts_local = self._resample_polyline(pts_local, self.resample_ds, self.max_samples)

        if len(pts_local) < 2:
            return None, goal_local

        th_ref = self._compute_reference_theta(pts_local)
        ref = np.column_stack((pts_local[:, 0], pts_local[:, 1], th_ref))
        return ref, goal_local

    # --------------------------------------------------
    # Band construction and optimization
    # --------------------------------------------------
    def _build_initial_band(self, ref: np.ndarray):
        #This initializes the timed band from the reference path.
        n = len(ref)
        x = ref[:, 0].copy()
        y = ref[:, 1].copy()
        th = ref[:, 2].copy()
        dt = np.full(max(n - 1, 1), self.min_dt, dtype=np.float64)

        for i in range(n - 1):
            ds = float(np.linalg.norm(ref[i + 1, :2] - ref[i, :2]))
            dth = abs(wrap_to_pi(ref[i + 1, 2] - ref[i, 2]))
            dt_v = ds / max(self.v_nominal, self.eps)
            dt_vmin = ds / max(self.max_vel_x, self.eps)
            dt_wmin = dth / max(self.max_vel_theta, self.eps)
            dt[i] = clamp(max(self.min_dt, dt_v, dt_vmin, dt_wmin), self.min_dt, self.max_dt)

        x[0] = 0.0
        y[0] = 0.0
        th[0] = 0.0
        return {'x': x, 'y': y, 'th': th, 'dt': dt, 'ref': ref}

    def _obstacle_force(self, pxy: np.ndarray):
        #This computes a repulsive force from static obstacles around one node.
        if self.obstacles_base.size == 0:
            return np.zeros(2, dtype=np.float64)

        delta = pxy[None, :] - self.obstacles_base
        dist = np.linalg.norm(delta, axis=1)
        mask = (dist > self.eps) & (dist < self.obstacle_influence_dist)

        if not np.any(mask):
            return np.zeros(2, dtype=np.float64)

        d = dist[mask]
        q = delta[mask]

        gain = (1.0 / np.maximum(d, self.eps) - 1.0 / self.obstacle_influence_dist) / np.maximum(d * d, self.eps)
        gain = gain[:, None]
        force = np.sum(gain * (q / np.maximum(d[:, None], self.eps)), axis=0)
        return force

    def _optimize_band(self, band):
        #This performs a small number of gradient-like iterations over x, y, theta, dt.
        x = band['x']
        y = band['y']
        th = band['th']
        dt = band['dt']
        ref = band['ref']
        n = len(x)

        if n < 2:
            return band

        for _ in range(self.max_iterations):
            for i in range(1, n - 1):
                p = np.array([x[i], y[i]], dtype=np.float64)
                p_prev = np.array([x[i - 1], y[i - 1]], dtype=np.float64)
                p_next = np.array([x[i + 1], y[i + 1]], dtype=np.float64)
                p_ref = ref[i, :2]

                #This pulls the node toward the reference path.
                f_path = (p_ref - p)

                #This smooths the band by attracting the node to the midpoint of its neighbors.
                f_smooth = 0.5 * (p_prev + p_next) - p

                #This pushes the node away from static obstacles.
                f_obs = self._obstacle_force(p)

                #This penalizes lateral motion between consecutive nodes for non-holonomic behavior.
                t_prev = np.array([math.cos(th[i - 1]), math.sin(th[i - 1])], dtype=np.float64)
                n_prev = np.array([-t_prev[1], t_prev[0]], dtype=np.float64)
                lateral_prev = float(np.dot((p - p_prev), n_prev))
                f_nh = -lateral_prev * n_prev

                total_force = (
                    self.w_path * f_path +
                    self.w_smooth * f_smooth +
                    self.w_obstacle * f_obs +
                    self.w_kinematics_nh * f_nh
                )

                p = p + self.alpha_xy * total_force
                x[i] = float(p[0])
                y[i] = float(p[1])

            #This updates orientation to align with local tangent and reference heading.
            for i in range(1, n - 1):
                seg = np.array([x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]], dtype=np.float64)
                seg_norm = float(np.linalg.norm(seg))
                if seg_norm > self.eps:
                    th_seg = math.atan2(seg[1], seg[0])
                else:
                    th_seg = th[i]

                e_seg = wrap_to_pi(th_seg - th[i])
                e_ref = wrap_to_pi(ref[i, 2] - th[i])
                th[i] = wrap_to_pi(th[i] + self.alpha_theta * (self.w_kinematics_nh * e_seg + self.w_theta_ref * e_ref))

            th[0] = 0.0
            th[-1] = wrap_to_pi(th[-1])

            #This updates timing to satisfy velocity/turn-rate constraints while keeping total time small.
            for i in range(n - 1):
                ds = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
                dth = abs(wrap_to_pi(th[i + 1] - th[i]))
                dt_vmin = ds / max(self.max_vel_x, self.eps)
                dt_wmin = dth / max(self.max_vel_theta, self.eps)
                dt_fast = ds / max(self.v_nominal, self.eps)
                dt_target = max(self.min_dt, dt_vmin, dt_wmin, 0.50 * dt_fast)

                if i == 0:
                    dt_smooth = dt_target
                elif i == (n - 2):
                    dt_smooth = dt[i - 1]
                else:
                    dt_smooth = 0.5 * (dt[i - 1] + dt[i + 1])

                blended = (
                    self.w_time * dt_target +
                    self.w_dt_smooth * dt_smooth
                ) / max(self.w_time + self.w_dt_smooth, self.eps)

                dt[i] = clamp((1.0 - self.alpha_dt) * dt[i] + self.alpha_dt * blended, self.min_dt, self.max_dt)

        band['x'] = x
        band['y'] = y
        band['th'] = th
        band['dt'] = dt
        return band

    # --------------------------------------------------
    # Command extraction
    # --------------------------------------------------
    def _limit_by_acceleration(self, v_cmd: float, w_cmd: float):
        #This enforces acceleration limits with respect to current odometry velocities.
        dt_ctrl = 1.0 / max(self.control_rate_hz, 1.0)
        dv_max = self.acc_lim_x * dt_ctrl
        dw_max = self.acc_lim_theta * dt_ctrl

        v_lo = self.current_v - dv_max
        v_hi = self.current_v + dv_max
        w_lo = self.current_w - dw_max
        w_hi = self.current_w + dw_max

        v_cmd = clamp(v_cmd, v_lo, v_hi)
        w_cmd = clamp(w_cmd, w_lo, w_hi)
        return v_cmd, w_cmd

    def _extract_cmd_from_band(self, band):
        #This extracts the first feasible control command from the optimized band.
        x = band['x']
        y = band['y']
        th = band['th']
        dt = band['dt']

        if len(x) < 2 or len(dt) < 1:
            return 0.0, 0.0

        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        ds = math.hypot(dx, dy)
        dth = wrap_to_pi(float(th[1] - th[0]))
        dt0 = max(float(dt[0]), self.eps)

        v_cmd = ds / dt0
        if dx < 0.0 and (not self.allow_reverse):
            v_cmd = 0.0

        w_cmd = dth / dt0

        #This optionally reduces speed on high local curvature to improve stability.
        if ds > self.eps and v_cmd > self.eps:
            kappa = abs(2.0 * dy / max(ds * ds, self.eps))
            v_curve = math.sqrt(self.max_lateral_accel / max(kappa, self.eps)) if kappa > self.eps else self.max_vel_x
            v_cmd = min(v_cmd, v_curve)

        v_cmd = clamp(v_cmd, 0.0 if not self.allow_reverse else -self.max_vel_x, self.max_vel_x)
        w_cmd = clamp(w_cmd, -self.max_vel_theta, self.max_vel_theta)
        v_cmd, w_cmd = self._limit_by_acceleration(v_cmd, w_cmd)
        return v_cmd, w_cmd

    # --------------------------------------------------
    # Goal logic
    # --------------------------------------------------
    def _goal_behavior(self, goal_local):
        #This handles stopping or final yaw alignment near the goal.
        if goal_local is None:
            return False

        gx, gy, gyaw = goal_local
        d_goal = math.hypot(gx, gy)
        yaw_err = wrap_to_pi(gyaw)

        if d_goal > self.goal_tolerance:
            return False

        if abs(yaw_err) <= self.yaw_goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info('Goal reached | distance and yaw within tolerance.')
            self.goal_reached = True
            self._stop()
            return True

        w_cmd = clamp(self.goal_yaw_kp * yaw_err, -self.max_vel_theta, self.max_vel_theta)
        _, w_cmd = self._limit_by_acceleration(0.0, w_cmd)
        self.goal_reached = False
        self._publish_cmd(0.0, w_cmd)
        return True

    # --------------------------------------------------
    # Main control loop
    # --------------------------------------------------
    def on_timer(self):
        #This runs the full TEB local planning cycle.
        if not self.has_path:
            self._stop()
            return

        ref, goal_local = self._build_local_reference()
        if ref is None or len(ref) < 2:
            self._stop()
            return

        if self._goal_behavior(goal_local):
            return

        band = self._build_initial_band(ref)
        band = self._optimize_band(band)
        self.last_band = band

        v_cmd, w_cmd = self._extract_cmd_from_band(band)
        self._publish_cmd(v_cmd, w_cmd)
        self._publish_band_path(band)


def main(args=None):
    rclpy.init(args=args)
    node = TEBStaticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()