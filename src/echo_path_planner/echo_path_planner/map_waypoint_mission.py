#!/usr/bin/env python3
"""
MissionManager: waypoints → /get_plan → /planned_path
Con RDP en ARA*, el path ya llega simplificado. Solo concatenar y publicar.
"""
import copy
import os
import select
import sys
import threading

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from tf2_ros import Buffer, TransformListener, TransformException


class SimpleMissionManager(Node):

    def __init__(self):
        super().__init__("simple_mission_manager")
        self._init_params()

        # ROS
        viz_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.goal_sub = self.create_subscription(
            PoseStamped, self.t_goal, self._on_goal, 10)
        self.path_pub = self.create_publisher(Path, self.t_path, 10)
        self.viz_pub = self.create_publisher(Path, self.t_viz, viz_qos)
        self.plan_cli = self.create_client(GetPlan, self.srv_name)
        self.tf_buf = Buffer()
        self.tf_lis = TransformListener(self.tf_buf, self)

        # Estado
        self.goals, self.goal_ids = [], []
        self.lock = threading.Lock()
        self._last_path = None

        # Republish
        if self.repub_on:
            self.create_timer(self.repub_sec, self._republish)

        # Consola
        self._alive = True
        threading.Thread(target=self._console, daemon=True).start()

        self.get_logger().info(
            f"MissionManager listo | ctrl={self.t_path} "
            f"viz={self.t_viz} srv={self.srv_name}")

    # ── Parámetros ──────────────────────────────────────

    def _init_params(self):
        d = self.declare_parameter
        d("frames.global_frame", "map")
        d("frames.base_frame", "base_link")
        d("topics.input_goal", "/goal_pose")
        d("topics.planned_path", "/planned_path")
        d("topics.full_path", "/mission_full_path")
        d("plan_service", "/get_plan")
        d("fixed_waypoints.file", "config/fixed_waypoints.yaml")
        d("mission.return_to_first_at_end", False)
        d("mission.allow_repeated_laps", True)
        d("republish.enabled", True)
        d("republish.period_sec", 1.0)
        d("republish.log", False)

        p = lambda n: self.get_parameter(n).value
        self.frame = p("frames.global_frame")
        self.base = p("frames.base_frame")
        self.t_goal = p("topics.input_goal")
        self.t_path = p("topics.planned_path")
        self.t_viz = p("topics.full_path")
        self.srv_name = p("plan_service")
        self.wp_file = self._find_file(p("fixed_waypoints.file"))
        self.ret_first = p("mission.return_to_first_at_end")
        self.allow_laps = p("mission.allow_repeated_laps")
        self.repub_on = p("republish.enabled")
        self.repub_sec = max(0.1, float(p("republish.period_sec")))
        self.repub_log = p("republish.log")

    # ── Goals ───────────────────────────────────────────

    def _on_goal(self, msg: PoseStamped):
        g = self._fix_pose(msg)
        if g.header.frame_id != self.frame:
            self.get_logger().warn(
                f"Goal frame '{g.header.frame_id}' ≠ '{self.frame}'")
            return
        with self.lock:
            self.goals.append(g)
            self.goal_ids.append(f"G{len(self.goals)}")
            n = len(self.goals)
        pos = g.pose.position
        self.get_logger().info(f"Goal G{n}: ({pos.x:.2f}, {pos.y:.2f})")

    # ── Consola ─────────────────────────────────────────

    def _console(self):
        cmds = {
            "start":       self._cmd_start,
            "start_fixed": self._cmd_start_fixed,
            "load_fixed":  self._load_yaml,
            "list":        self._cmd_list,
            "clear":       self._cmd_clear,
            "stop":        self._cmd_stop,
        }
        while rclpy.ok() and self._alive:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
            except Exception:
                continue
            if not r:
                continue
            line = sys.stdin.readline().strip().lower()
            if not line:
                continue
            handler = cmds.get(line)
            if handler:
                handler()
            else:
                print("Comandos: " + ", ".join(cmds))

    def _cmd_start(self):
        with self.lock:
            if not self.goals:
                print("No goals"); return
            gs = copy.deepcopy(self.goals)
            ids = copy.deepcopy(self.goal_ids)

        try:
            laps = int(input("Laps?: "))
            assert laps > 0
        except (ValueError, EOFError, AssertionError):
            print("Inválido"); return

        seq, sids = self._build_seq(gs, ids, laps)
        self.get_logger().info(
            f"Misión: {len(seq)} goals, {laps} lap(s) "
            f"[{sids[0]} → {sids[-1]}]")

        path = self._precompute(seq)
        if not path or not path.poses:
            self.get_logger().error("✗ Precomputación falló")
            return

        self._publish(path)

    def _cmd_start_fixed(self):
        if self._load_yaml():
            self._cmd_start()

    def _cmd_list(self):
        with self.lock:
            pairs = list(zip(self.goal_ids, self.goals))
        if not pairs:
            print("No goals"); return
        for gid, g in pairs:
            print(f"  {gid}: ({g.pose.position.x:.2f}, "
                  f"{g.pose.position.y:.2f})")

    def _cmd_clear(self):
        with self.lock:
            self.goals.clear()
            self.goal_ids.clear()
        print("Cleared")

    def _cmd_stop(self):
        self._last_path = None
        empty = self._empty_path()
        self.path_pub.publish(empty)
        self.viz_pub.publish(empty)
        print("Stop")

    # ── Misión ──────────────────────────────────────────

    def _build_seq(self, goals, ids, laps):
        if not self.allow_laps:
            laps = 1
        sg, si = [], []
        for _ in range(laps):
            sg.extend(copy.deepcopy(goals))
            si.extend(copy.deepcopy(ids))
        if self.ret_first and len(goals) > 1:
            sg.append(copy.deepcopy(goals[0]))
            si.append(ids[0])
        return sg, si

    def _precompute(self, goals):
        start = self._robot_pose()
        if not start:
            self.get_logger().error("Sin pose del robot")
            return None

        segments, cur = [], start
        for i, goal in enumerate(goals):
            goal = self._fix_pose(goal)
            self.get_logger().info(
                f"  Seg {i+1}/{len(goals)}: "
                f"({cur.pose.position.x:.1f},"
                f"{cur.pose.position.y:.1f}) → "
                f"({goal.pose.position.x:.1f},"
                f"{goal.pose.position.y:.1f})")

            seg = self._call_plan(cur, goal)
            if not seg or not seg.poses:
                self.get_logger().error(f"  ✗ Seg {i+1} falló")
                return None

            self.get_logger().info(f"  ✓ {len(seg.poses)} poses")
            segments.append(seg)
            cur = goal

        return self._concat(segments)

    def _call_plan(self, start, goal, timeout=8.0):
        if not self.plan_cli.service_is_ready():
            if not self.plan_cli.wait_for_service(timeout):
                self.get_logger().warn("/get_plan no disponible")
                return None

        req = GetPlan.Request()
        req.start, req.goal, req.tolerance = start, goal, 0.0

        future = self.plan_cli.call_async(req)
        done, box = threading.Event(), {}

        def cb(f):
            try:
                box["r"] = f.result()
            except Exception as e:
                box["e"] = e
            done.set()

        future.add_done_callback(cb)

        if not done.wait(timeout):
            self.get_logger().warn("Timeout /get_plan")
            return None
        if "e" in box:
            self.get_logger().error(f"Error: {box['e']}")
            return None

        plan = box["r"].plan
        return plan if plan.poses else None

    # ── Publicación ─────────────────────────────────────

    def _publish(self, path: Path):
        kb = len(path.poses) * 160 / 1024

        self.path_pub.publish(path)
        self._last_path = copy.deepcopy(path)
        self.get_logger().info(
            f"✓ {self.t_path}: {len(path.poses)} poses (~{kb:.0f} KB)")

        self.viz_pub.publish(self._empty_path())
        self.viz_pub.publish(path)
        self.get_logger().info(
            f"✓ {self.t_viz}: {len(path.poses)} poses")

    def _republish(self):
        if self._last_path and self._last_path.poses:
            self.path_pub.publish(self._last_path)
            if self.repub_log:
                self.get_logger().info(
                    f"Republish: {len(self._last_path.poses)} poses")

    # ── YAML ────────────────────────────────────────────

    def _load_yaml(self):
        if not os.path.exists(self.wp_file):
            self.get_logger().warn(f"No existe: {self.wp_file}")
            return False

        try:
            with open(self.wp_file, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"YAML error: {e}")
            return False

        if isinstance(data, dict):
            wps = data.get("waypoints", [])
            fr = data.get("frame_id", self.frame)
        elif isinstance(data, list):
            wps, fr = data, self.frame
        else:
            self.get_logger().warn("YAML inválido")
            return False

        if fr != self.frame:
            self.get_logger().warn(f"Frame '{fr}' ≠ '{self.frame}'")
            return False

        goals, ids = [], []
        stamp = self.get_clock().now().to_msg()
        for i, wp in enumerate(wps):
            try:
                g = PoseStamped()
                g.header.frame_id = self.frame
                g.header.stamp = stamp
                g.pose.position.x = float(wp["x"])
                g.pose.position.y = float(wp["y"])
                g.pose.position.z = float(wp.get("z", 0.0))
                g.pose.orientation.w = 1.0
                goals.append(g)
                ids.append(str(wp.get("id", f"G{i+1}")))
            except Exception as e:
                self.get_logger().warn(f"WP {i}: {e}")
                return False

        if not goals:
            self.get_logger().warn("Sin waypoints")
            return False

        with self.lock:
            self.goals, self.goal_ids = goals, ids
        self.get_logger().info(f"Cargados {len(goals)} waypoints")
        return True

    # ── Utils ───────────────────────────────────────────

    def _robot_pose(self):
        try:
            tf = self.tf_buf.lookup_transform(
                self.frame, self.base, rclpy.time.Time(seconds=0))
        except TransformException as e:
            self.get_logger().warn(f"TF: {e}")
            return None
        p = PoseStamped()
        p.header.frame_id = self.frame
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = tf.transform.translation.x
        p.pose.position.y = tf.transform.translation.y
        p.pose.position.z = tf.transform.translation.z
        p.pose.orientation = tf.transform.rotation
        return p

    def _fix_pose(self, pose):
        p = copy.deepcopy(pose)
        if not p.header.frame_id:
            p.header.frame_id = self.frame
        p.header.stamp = self.get_clock().now().to_msg()
        q = p.pose.orientation
        if abs(q.x) + abs(q.y) + abs(q.z) + abs(q.w) < 1e-6:
            p.pose.orientation.w = 1.0
        return p

    def _concat(self, segments):
        full = Path()
        full.header.frame_id = self.frame
        full.header.stamp = self.get_clock().now().to_msg()
        for i, seg in enumerate(segments):
            start = 1 if i > 0 else 0
            for pose in seg.poses[start:]:
                p = copy.deepcopy(pose)
                p.header = copy.deepcopy(full.header)
                full.poses.append(p)
        return full

    def _empty_path(self):
        p = Path()
        p.header.frame_id = self.frame
        p.header.stamp = self.get_clock().now().to_msg()
        return p

    def _find_file(self, path_str):
        s = os.path.expanduser(os.path.expandvars(str(path_str)))
        if os.path.isabs(s):
            return s
        cwd = os.path.abspath(s)
        if os.path.exists(cwd):
            return cwd
        try:
            from ament_index_python.packages import \
                get_package_share_directory
            c = os.path.join(
                get_package_share_directory("echo_path_planner"),
                "config", os.path.basename(s))
            if os.path.exists(c):
                return c
        except Exception:
            pass
        return cwd

    def destroy_node(self):
        self._alive = False
        super().destroy_node()


def main():
    rclpy.init()
    node = SimpleMissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()