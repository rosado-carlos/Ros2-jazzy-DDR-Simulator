#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped

class AEBNode(Node):    #This class implement an AEB (Automatic Emergency Brake) based on LiDAR TTC.

    def __init__(self):
        super().__init__('aeb')

        # ---------- parameters ----------
        #This declare ROS parameters with default values (can be changed from launch/CLI).
        self.declare_parameter("ttc_threshold", 1)
        self.declare_parameter("min_distance", 0.6)
        self.declare_parameter("robot_radius", 0.3)

        #This are angle parameters in degrees that will be converted to radians.
        self.declare_parameter("front_angle_deg", 20.0)
        self.declare_parameter("velocity_sector_deg", 35.0)

        #This are thresholds for speed estimation and TTC computation stability.
        self.declare_parameter("min_speed", 0.05)
        self.declare_parameter("eps_closing", 0.1)
        self.declare_parameter("max_speed", 5.0)

        # ---------- status variables ----------
        self.lock = False   #This is for a frontal block and rotative block (here only stop, but name kept).
        self.forward_Block = False      #This is for a frontal block latch (keeps blocking forward until out of distance).

        # ---------- tresholds ----------
        self.ttc_treshold = float(self.get_parameter("ttc_threshold").value)    #This is the treshold for the minimum ttc.
        self.min_distance = float(self.get_parameter("min_distance").value)     #This is the treshold for the minimum distance allowed when TTC was activated.

        # ---- lidar geometry ----------
        self.front_angle = np.deg2rad(float(self.get_parameter("front_angle_deg").value))   #This is the front angle of vision that will be consider for TTC ±front_angle.
        self.v_sector = np.deg2rad(float(self.get_parameter("velocity_sector_deg").value))  #This is the angle range that will be use to estimate the ego velocity ±v_sector.
        self.radius = float(self.get_parameter("robot_radius").value)   #This is the radius of the circunference that glob the car (inflation for TTC).
        self.min_speed = float(self.get_parameter("min_speed").value)   #This is the minimum speed magnitude below which speed is treated as zero.
        self.eps_closing = float(self.get_parameter("eps_closing").value)   #This is the minimum closing speed for a ray to be considered for TTC.
        self.max_speed = float(self.get_parameter("max_speed").value)   #This is the maximum acceptable speed candidate from lidar differencing.

        # ---- lidar state for velocity estimation ----
        self.prev_ranges = None     #This store the previous ranges (last scan) for differencing.
        self.prev_stamp = None      #This store the previous stamp in seconds for dt computation.

        # ---------- subscriptors ----------
        self.scan_sub = self.create_subscription(LaserScan,'/scan',self.scan_callback,10)   

        self.cmdjoy_sub = self.create_subscription(TwistStamped,'/cmd_vel_joy',self.cmd_control_callback,10)

        self.cmdjoy_sub = self.create_subscription(TwistStamped,'/cmd_vel_ctrl',self.cmdjoy_callback,10)

        # ---------- publisher ----------
        self.cmd_pub = self.create_publisher(TwistStamped,'/cmd_vel_safe',10)

        #Test
        self.d_min = 0.0

    # --------------------------------------------------
    # Small helpers for lidar indexing and validation
    # --------------------------------------------------
    def _angle_to_index(self, scan_msg, a):     #This convert an angle (rad) to the nearest scan index.
        return int(round((a - scan_msg.angle_min) / scan_msg.angle_increment))

    def _valid_range(self, scan_msg, r):    #This validate a range: finite and inside sensor min/max.
        return np.isfinite(r) and (scan_msg.range_min < r < scan_msg.range_max)

    # --------------------------------------------------
    # Stop publisher
    # --------------------------------------------------
    def _stop(self):     #This publish a stop command (linear and angular 0) to /cmd_vel_safe.
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = ""
        out.twist.linear.x = 0.0
        out.twist.angular.z = 0.0
        self.cmd_pub.publish(out)

    # --------------------------------------------------
    # TTC computation
    # --------------------------------------------------
    def _ttc_calculus(self, v, d):  #This compute the minimum TTC using the forward sector ranges and angles.

        ranges_front, thetas_front = d      #This unpack the tuple (ranges array, angles array) for the forward sector.

        if v <= 0.0 or ranges_front.size == 0:  #This means we are not moving forward or no valid ranges, so no collision risk by TTC.
            return float('inf')

        v_closing = v * np.cos(thetas_front)    #This project the ego forward speed into each ray direction (closing speed per ray).
        mask = v_closing > self.eps_closing     #This keep only rays that actually have meaningful closing speed.

        if not np.any(mask):    #This means no ray is being approached fast enough, so TTC is infinite.
            return float('inf')

        dist = ranges_front[mask] - self.radius     #This subtract robot radius (inflation) so TTC is with respect to the robot boundary.

        if np.any(dist <= 0.0):     #This means we are already inside the inflated radius: collision now.
            return 0.0

        ttc = dist / v_closing[mask]    #This compute TTC per ray.
        return float(np.min(ttc)) if ttc.size else float('inf')     #This return the minimum TTC across the forward sector.

    # --------------------------------------------------
    # Forward block logic for joystick callback
    # --------------------------------------------------
    def _forward_Block_(self, joy_msg):     #This block forward motion when forward_Block is latched ON.

        if self.forward_Block:      #This means forward movement should be disallowed.
            if joy_msg.twist.linear.x > 0 or joy_msg.twist.angular.z !=0.0:  #This means user is trying to move forward.
                self._stop()    #This apply a stop override to prevent forward motion.
                self.get_logger().warn("FWD_BLOCK | cmd forward rejected (vx>0). Move back or increase distance.")#This log only when we actively reject a forward command.
            else:   #This means the command is not forward (reverse or zero).
                self.get_logger().debug("FWD_BLOCK | cmd allowed (vx<=0).")     #This debug log avoids spamming info logs during the block state.

    # --------------------------------------------------
    # Extract ranges/angles for a sector
    # --------------------------------------------------
    def _sector_ranges_and_angles(self, lidar_msg, ranges, a0, a1):     #This extract ranges and corresponding angles between [a0, a1] and filter invalid data.

        i0 = max(0, self._angle_to_index(lidar_msg, a0))    #This compute the start index and clamp to 0.
        i1 = min(len(ranges) - 1, self._angle_to_index(lidar_msg, a1))      #This compute the end index and clamp to n-1.

        if i1 < i0:     #This means invalid sector window, return empty arrays.
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        seg = ranges[i0:i1+1]   #This slice the ranges for the sector.
        idxs = np.arange(i0, i1 + 1)    #This create the indices for the sector.
        thetas = lidar_msg.angle_min + idxs * lidar_msg.angle_increment     #This compute the absolute angle for each sector index.
        mask = np.isfinite(seg) & (lidar_msg.range_min < seg) & (seg < lidar_msg.range_max)     #This mask out invalid and out-of-range measurements.
        return seg[mask], thetas[mask]      #This return filtered ranges and their matching angles.

    # --------------------------------------------------
    # Kinematic state by lidar (vx estimate, dx sector, self.d_min)
    # --------------------------------------------------

    def _kin_state_by_lidar(self, lidar_msg: LaserScan):    #This estimate forward ego velocity (vx) and extract forward sector for TTC.

        st = lidar_msg.header.stamp     #This read the ROS stamp from the message.
        t = float(st.sec) + 1e-9 * float(st.nanosec)    #This convert stamp to seconds.
        ranges = np.asarray(lidar_msg.ranges, dtype=np.float64)     #This convert ranges list to numpy array for faster operations.
        n = len(ranges)     #This store number of rays.
        #This always compute the forward sector arrays (even if vx cannot be computed yet).
        ranges_front, thetas_front = self._sector_ranges_and_angles(lidar_msg, ranges, -self.front_angle, +self.front_angle)
        self.d_min = float(np.min(ranges_front)) if ranges_front.size else float('inf')      #This compute the minimum distance in the forward sector.

        #This handle the first scan (no previous scan to estimate velocity).
        if self.prev_ranges is None or self.prev_stamp is None:
            self.prev_ranges = ranges.copy()    #This store current scan as previous.
            self.prev_stamp = t     #This store current time as previous.
            return 0.0, (ranges_front, thetas_front), self.d_min     #This return vx=0 and the front sector data.
        
        dt = t - self.prev_stamp    #This compute dt between scans.

        if dt <= 1e-4 or dt > 0.5:      #This avoid division by zero and ignore stale dt.
            self.prev_ranges = ranges.copy()    #This refresh previous scan.
            self.prev_stamp = t     #This refresh previous time.
            return 0.0, (ranges_front, thetas_front), self.d_min     #This return vx=0 for unstable dt.

        prev = self.prev_ranges     #This read previous scan ranges.

        # ----- estimate vx in ±v_sector -----
        i0v = max(0, self._angle_to_index(lidar_msg, -self.v_sector))   #This compute start index for velocity sector.
        i1v = min(n - 1, self._angle_to_index(lidar_msg, +self.v_sector))   #This compute end index for velocity sector.
        idxs_v = np.arange(i0v, i1v + 1)    #This list the indices in the velocity sector.
        thetas_v = lidar_msg.angle_min + idxs_v * lidar_msg.angle_increment     #This compute angles for each index.
        c = np.cos(thetas_v)    #This compute cosines for projection into forward axis.

        v_candidates = []
        #This will store candidate speed estimates.

        #This loop over rays in the velocity sector.
        for k, i in enumerate(idxs_v):
            r = ranges[i]   #This is the current range.
            rp = prev[i] if i < len(prev) else np.inf       #This is the previous range at same index.

            if not (self._valid_range(lidar_msg, r) and self._valid_range(lidar_msg, rp)):      #This skip invalid measurements.
                continue

            if abs(c[k]) < 0.3:     #This skip near lateral rays to avoid dividing by very small projection.
                continue

            vi = (rp - r) / (dt * c[k])     #This estimate forward speed from range change divided by dt and cos(theta).

            if 0.0 <= vi <= self.max_speed:     #This keep only reasonable positive speeds.
                v_candidates.append(vi)

        vx = float(np.median(v_candidates)) if v_candidates else 0.0    #This take the median to reduce outliers.

        if (not np.isfinite(vx)) or abs(vx) < self.min_speed:   #This clamp very small or invalid speed estimates to zero.
            vx = 0.0

        # update memory
        self.prev_ranges = ranges.copy()
        #This save current scan for next velocity estimation.

        self.prev_stamp = t     #This save current time for next dt.
        return vx, (ranges_front, thetas_front), self.d_min  #This return the speed, the TTC sector data, and the min distance.

    # --------------------------------------------------
    # LiDAR callback
    # --------------------------------------------------

    def scan_callback(self, msg):

        vx, dx, self.d_min = self._kin_state_by_lidar(msg)   #This return the estimated ego forward speed vx, the forward sector data dx, and self.d_min.
        ttc_min = self._ttc_calculus(vx, dx)    #This compute the minimum TTC using vx and the forward sector.
        prev_lock = self.lock   #This store previous lock state to detect transitions.

        if self.d_min > 1.3:
            self.ttc_treshold = 0.3*float(self.get_parameter("ttc_threshold").value)
        elif self.d_min > 1:
            self.ttc_treshold = 0.5*float(self.get_parameter("ttc_threshold").value)
        elif self.d_min > 0.7:
            self.ttc_treshold = float(self.get_parameter("ttc_threshold").value)
        elif self.d_min > 0.5:
            self.ttc_treshold = 1*float(self.get_parameter("ttc_threshold").value)
        else:
            self.ttc_treshold = 1.7*float(self.get_parameter("ttc_threshold").value)

        self.lock = (self.d_min <= 1.5) and (ttc_min < self.ttc_treshold)

        #This log only on LOCK transitions.
        if self.lock and not prev_lock:
            self.get_logger().warn(
                f"LOCK ON  | TTCmin={ttc_min:.2f}s < {self.ttc_treshold:.2f}s | "
                f"vx={vx:.2f} m/s | dmin={self.d_min:.2f} m | "f"min_dist={self.min_distance:.2f} m")   #This is the risk entry log with enough context.
        elif (not self.lock) and prev_lock:
            self.get_logger().info(
                f"LOCK OFF | TTCmin={ttc_min:.2f}s >= {self.ttc_treshold:.2f}s | "f"vx={vx:.2f} m/s | dmin={self.d_min:.2f} m")      #This is the risk exit log.

        if self.lock:   #This means risk is present by TTC, so we must brake.
            self._stop()

        # --- FORWARD_BLOCK transitions ---
        prev_fb = self.forward_Block    #This store previous forward block state to detect transitions.

        if self.lock and (self.d_min < self.min_distance):   #This latch forward_Block ON when we are locked and too close.
            self.forward_Block = True
        elif self.forward_Block and (self.d_min >= self.min_distance):   #This release forward_Block only when distance is safe again.
            self.forward_Block = False

        if self.forward_Block and not prev_fb:self.get_logger().warn(
                f"FWD_BLOCK ON  | dmin={self.d_min:.2f} m < {self.min_distance:.2f} m | "f"Reason: too close after TTC lock")    #This indicates the latch is now active.
        elif (not self.forward_Block) and prev_fb:self.get_logger().info(
            f"FWD_BLOCK OFF | dmin={self.d_min:.2f} m >= {self.min_distance:.2f} m | "f"Reason: distance recovered")     #This indicates latch released.

    # --------------------------------------------------
    # Joystick callback
    # --------------------------------------------------

    def cmdjoy_callback(self, msg):
        self._forward_Block_(msg)   #This enforces forward blocking when forward_Block is active.
    
    def cmd_control_callback(self, msg):
        self._forward_Block_(msg)   #This enforces forward blocking when forward_Block is active.


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
