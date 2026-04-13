#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray

class LidarDataNode(Node):    #This class implement an AEB (Automatic Emergency Brake) based on LiDAR TTC.

    def __init__(self):
        super().__init__('lidar_data')

        # ---------- parameters ----------
        #This are angle parameters in degrees that will be converted to radians.
        self.declare_parameter("front_angle_deg", 20.0)
        self.declare_parameter("velocity_sector_deg", 30.0)

        #This are thresholds for speed estimation
        self.declare_parameter("min_speed", 0.05)
        self.declare_parameter("max_speed", 5.0)

        # ---- lidar geometry ----------
        self.front_angle = np.deg2rad(float(self.get_parameter("front_angle_deg").value))   #This is the front angle of vision that will be consider for TTC ±front_angle.
        self.v_sector = np.deg2rad(float(self.get_parameter("velocity_sector_deg").value))  #This is the angle range that will be use to estimate the ego velocity ±v_sector.
        self.min_speed = float(self.get_parameter("min_speed").value)   #This is the minimum speed magnitude below which speed is treated as zero.
        self.max_speed = float(self.get_parameter("max_speed").value)   #This is a reasonable maximum speed for filtering outliers in velocity estimation (can be adjusted based on the robot's capabilities).
        #
        #  ---- lidar state for velocity estimation ----
        self.prev_ranges = None     #This store the previous ranges (last scan) for differencing.
        self.prev_stamp = None      #This store the previous stamp in seconds for dt computation.

        # ---------- subscriptors ----------
        self.scan_sub = self.create_subscription(LaserScan,'/scan',self.scan_callback,1)   

        # ---------- publisher ----------
        self.vx_pub = self.create_publisher(Float32, '/lidar/vx', 10) #This publish the estimated forward velocity based on LiDAR.
        self.dmin_pub = self.create_publisher(Float32, '/lidar/d_min', 10) #This publish the minimum distance in the front sector based on LiDAR.
        self.front_pub = self.create_publisher(Float32MultiArray, '/lidar/front_ranges', 10) #This publish the ranges in the front sector as a Float32MultiArray for potential use in TTC computation or other decision logic.

        #Test
        self.d_min = 0.0
        self.vx_filt = 0.0
        self.alpha = 0.5


    # Small helpers for lidar indexing and validation
    def _angle_to_index(self, scan_msg, a):     #This convert an angle (rad) to the nearest scan index.
        return int(round((a - scan_msg.angle_min) / scan_msg.angle_increment))

    def _valid_range(self, scan_msg, r):    #This validate a range: finite and inside sensor min/max.
        return np.isfinite(r) and (scan_msg.range_min < r < scan_msg.range_max)

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
        
        vr_list = []
        cos_list = []

        #This loop over rays in the velocity sector.
        for k, i in enumerate(idxs_v):
            r = ranges[i]   #This is the current range.
            rp = prev[i] if i < len(prev) else np.inf       #This is the previous range at same index.

            if not (self._valid_range(lidar_msg, r) and self._valid_range(lidar_msg, rp)):      #This skip invalid measurements.
                continue

            ctheta = np.cos(thetas_v[k])

            if abs(ctheta) < 0.5:     #This skip near lateral rays to avoid dividing by very small projection.
                continue

            if abs(r - rp) > 1.5:     #This skip rays with large range change that are likely outliers or dynamic objects.
                continue

            vr = (rp - r) / dt     #This estimate radial speed

            if abs(vr) > self.max_speed:
                continue

            vr_list.append(vr)
            cos_list.append(ctheta)

        if len(vr_list) < 5:
            vx = 0.0
        else:
            A = np.array(cos_list).reshape(-1, 1)
            b = np.array(vr_list)

            vx = 0.0

            vx_ls = float(np.linalg.lstsq(A, b, rcond=None)[0][0])
            if not np.isfinite(vx_ls):
                vx_ls = 0.0
            vx_candidates = b / A.flatten()
            vx_median = float(np.median(vx_candidates))

            vx = 0.7 * vx_ls + 0.3 * vx_median   #This combine the least squares estimate and the median of individual estimates to get a more robust velocity estimate.

        if (not np.isfinite(vx)) or abs(vx) < self.min_speed:   #This clamp very small or invalid speed estimates to zero.
            vx = 0.0

        # update memory
        self.prev_ranges = ranges
        #This save current scan for next velocity estimation.

        self.prev_stamp = t     #This save current time for next dt.
        return vx, (ranges_front, thetas_front), self.d_min  #This return the speed, the TTC sector data, and the min distance.

    # --------------------------------------------------
    # LiDAR callback
    # --------------------------------------------------

    def scan_callback(self, msg):

        vx, dx, self.d_min = self._kin_state_by_lidar(msg)   #This return the estimated ego forward speed vx, the forward sector data dx, and self.d_min.
        vx_filt_last = self.vx_filt    #This store the last filtered velocity before updating.
        alpha = 0.5 if abs(vx - vx_filt_last) > 0.5 else 0.2    #This adapt the filter alpha based on how much the velocity changed (more responsive to big changes, smoother for small changes).
        self.vx_filt = (1 - alpha) * vx_filt_last + alpha * vx    #This apply a simple low-pass filter to the velocity estimate to reduce noise (optional, can be removed if not desired).
        vx = self.vx_filt    #This use the filtered velocity for TTC computation and decision.
        # --- velocity ---
        vx_msg = Float32()
        vx_msg.data = float(vx)
        self.vx_pub.publish(vx_msg)

        # --- minimum distance ---
        dmin_msg = Float32()
        dmin_msg.data = float(self.d_min)
        self.dmin_pub.publish(dmin_msg)

        # --- front ranges ---
        ranges_front, _ = dx

        front_msg = Float32MultiArray()
        front_msg.data = ranges_front.tolist()
        self.front_pub.publish(front_msg)

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