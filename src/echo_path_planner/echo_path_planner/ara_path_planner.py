#!/usr/bin/env python3
"""
ARA* (Anytime Repairing A*) grid planner — ROS 2
Incluye simplificación Ramer-Douglas-Peucker al final de cada plan.
"""
import math
import heapq
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


@dataclass
class ARANode:
    x: int
    y: int
    g: float = float('inf')
    v: float = float('inf')
    parent: Optional[Tuple[int, int]] = None

    def __lt__(self, other):
        return False


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


class ARAPlannerNode(Node):

    def __init__(self):
        super().__init__('ara_path_planner')

        self._declare_params()
        self._read_params()
        self._setup_ros()

        # Estado interno
        self._map: Optional[OccupancyGrid] = None
        self._obstacles: Optional[np.ndarray] = None
        self._dist_cells: Optional[np.ndarray] = None
        self._grid = None

        self.get_logger().info("ARA* Planner listo, esperando mapa...")
        self.get_logger().info(f"Servicio activo en: {self._srv_name}")
        self.get_logger().info(
            f"RDP tolerance: {self._rdp_tol:.3f} m "
            f"(0 = desactivado)")

    # ═══════════════════════════════════════════════════════
    #  Parámetros
    # ═══════════════════════════════════════════════════════

    def _declare_params(self):
        d = self.declare_parameter

        # Topics / frames
        d('topics.map_topic', '/map')
        d('topics.goal_topic', '/goal_pose')
        d('topics.path_topic', '/ara_single_path')
        d('topics.debug_paths', '/ara_debug_paths')
        d('plan_service', '/get_plan')
        d('goal_source', 'mission')
        d('frames.base_frame', 'base_link')
        d('frames.global_frame', 'map')

        # Grilla
        d('geometry.occupied_threshold', 60)
        d('geometry.use_8_connected', True)
        d('geometry.inflate_radius', 0.3)
        d('geometry.treat_unknown_as_obstacle', True)
        d('geometry.snap_start_goal_to_free', True)
        d('geometry.snap_max_radius', 1.0)

        # ARA* core
        d('ara_core.epsilon_start', 2.8)
        d('ara_core.epsilon_decrease', 0.7)
        d('ara_core.time_limit_sec', 2.0)
        d('ara_core.heuristic_type', 'euclidean')

        # ══════════════════════════════════════════════════
        #  NUEVO: Simplificación RDP
        # ══════════════════════════════════════════════════
        d('path_simplify.tolerance_m', 0.03)

        # Debug
        d('debug.publish_all_paths', False)
        d('debug.log_service_requests', True)
        d('debug.publish_service_plans', False)

    def _read_params(self):
        p = lambda n: self.get_parameter(n).value

        self._heuristic_type = str(p('ara_core.heuristic_type')).lower()
        self._use_8_conn = bool(p('geometry.use_8_connected'))
        self._debug_mode = bool(p('debug.publish_all_paths'))
        self._log_svc = bool(p('debug.log_service_requests'))
        self._pub_svc_plans = bool(p('debug.publish_service_plans'))
        self._srv_name = str(p('plan_service'))
        self.goal_source = str(p('goal_source'))

        # NUEVO
        self._rdp_tol = float(p('path_simplify.tolerance_m'))

        # Pre-calcular movimientos
        straight = [(0, 1, 1.0), (0, -1, 1.0),
                     (1, 0, 1.0), (-1, 0, 1.0)]
        diagonal = [(1, 1, 1.4142), (-1, 1, 1.4142),
                     (1, -1, 1.4142), (-1, -1, 1.4142)]
        self._moves = straight + diagonal if self._use_8_conn else straight

    def _setup_ros(self):
        p = lambda n: self.get_parameter(n).value

        qos_map = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.map_sub = self.create_subscription(
            OccupancyGrid, p('topics.map_topic'), self._on_map, qos_map)
        self.goal_rviz_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self._on_goal_rviz, 10)
        self.goal_mission_sub = self.create_subscription(
            PoseStamped, '/mission_goal', self._on_goal_mission, 10)

        self.path_pub = self.create_publisher(
            Path, p('topics.path_topic'), 10)
        self.debug_pub = self.create_publisher(
            MarkerArray, p('topics.debug_paths'), 10)

        self.plan_srv = self.create_service(
            GetPlan, self._srv_name, self._on_get_plan)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    # ═══════════════════════════════════════════════════════
    #  Callbacks
    # ═══════════════════════════════════════════════════════

    def _on_map(self, msg: OccupancyGrid):
        W, H, res = msg.info.width, msg.info.height, msg.info.resolution
        self._map = msg

        grid = np.array(msg.data, dtype=np.int16).reshape((H, W))
        self._grid = grid

        occ_th = self.get_parameter(
            'geometry.occupied_threshold').value
        unk_obs = self.get_parameter(
            'geometry.treat_unknown_as_obstacle').value

        obstacles = grid >= occ_th
        if unk_obs:
            obstacles = np.logical_or(obstacles, grid == -1)

        dist = self._brushfire(obstacles)
        self._dist_cells = dist

        inflate_r = float(self.get_parameter(
            'geometry.inflate_radius').value)
        if inflate_r > 1e-6:
            cells = int(math.ceil(inflate_r / res))
            obstacles = np.logical_or(obstacles, dist <= cells)

        self._obstacles = obstacles
        self.get_logger().info(f'Mapa: {W}x{H}, res={res:.3f} m/px')

    def _on_goal_rviz(self, msg):
        if self.goal_source == "rviz":
            self._process_goal(msg)

    def _on_goal_mission(self, msg):
        if self.goal_source == "mission":
            self._process_goal(msg)

    def _process_goal(self, msg: PoseStamped):
        if self._map is None or self._obstacles is None:
            self.get_logger().warn("Sin mapa"); return

        start = self._robot_pose()
        if start is None:
            return

        path = self._plan(start, msg)
        if path:
            self.path_pub.publish(path)
            self.get_logger().info(
                f"Ruta publicada: {len(path.poses)} poses")
        else:
            self.get_logger().error("ARA* falló")

    def _on_get_plan(self, request, response):
        self.get_logger().info("Solicitud /get_plan recibida")

        if self._map is None or self._obstacles is None:
            self.get_logger().warn("Sin mapa, rechazando")
            return response

        start = request.start
        if not start.header.frame_id:
            self.get_logger().warn("Request sin start, usando TF...")
            start = self._robot_pose()
            if start is None:
                return response

        path = self._plan(start, request.goal)

        if path:
            response.plan = path
            if self._pub_svc_plans:
                self.path_pub.publish(path)
            self.get_logger().info(
                f"✓ Plan: {len(path.poses)} poses")
        else:
            self.get_logger().error("✗ ARA* no encontró ruta")

        return response

    # ═══════════════════════════════════════════════════════
    #  Auxiliares de grilla
    # ═══════════════════════════════════════════════════════

    def _brushfire(self, obstacles: np.ndarray) -> np.ndarray:
        H, W = obstacles.shape
        INF = np.iinfo(np.int32).max
        dist = np.full((H, W), INF, dtype=np.int32)

        q = deque()
        ys, xs = np.nonzero(obstacles)
        for y, x in zip(ys, xs):
            dist[y, x] = 0
            q.append((x, y))

        if not q:
            return dist

        nbr = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            x, y = q.popleft()
            nd = dist[y, x] + 1
            for dx, dy in nbr:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and dist[ny, nx] > nd:
                    dist[ny, nx] = nd
                    q.append((nx, ny))
        return dist

    def _world_to_map(self, x, y, x0, y0, res, W, H):
        ix = int(math.floor((x - x0) / res))
        iy = int(math.floor((y - y0) / res))
        if 0 <= ix < W and 0 <= iy < H:
            return ix, iy
        return None

    def _map_to_world(self, ix, iy, x0, y0, res):
        return x0 + (ix + 0.5) * res, y0 + (iy + 0.5) * res

    def _is_free(self, idx):
        x, y = idx
        H, W = self._obstacles.shape
        return 0 <= x < W and 0 <= y < H and not bool(self._obstacles[y, x])

    def _nearest_free(self, idx, max_r):
        x0, y0 = idx
        H, W = self._obstacles.shape
        if self._is_free(idx):
            return idx

        best, best_d2 = None, None
        for r in range(1, max_r + 1):
            xmin, xmax = max(0, x0 - r), min(W - 1, x0 + r)
            ymin, ymax = max(0, y0 - r), min(H - 1, y0 + r)

            cands = []
            for x in range(xmin, xmax + 1):
                cands.append((x, ymin))
                cands.append((x, ymax))
            for y in range(ymin + 1, ymax):
                cands.append((xmin, y))
                cands.append((xmax, y))

            for x, y in cands:
                if not self._obstacles[y, x]:
                    d2 = (x - x0) ** 2 + (y - y0) ** 2
                    if best is None or d2 < best_d2:
                        best, best_d2 = (x, y), d2

            if best is not None:
                return best
        return None

    def _neighbors(self, ix, iy, W, H):
        result = []
        for dx, dy, cost in self._moves:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < W and 0 <= ny < H and not self._obstacles[ny, nx]:
                if abs(dx) == 1 and abs(dy) == 1:
                    if self._obstacles[iy, nx] or self._obstacles[ny, ix]:
                        continue
                result.append((nx, ny, cost))
        return result

    def _robot_pose(self):
        gf = self.get_parameter('frames.global_frame').value
        bf = self.get_parameter('frames.base_frame').value
        try:
            tf = self.tf_buffer.lookup_transform(
                gf, bf, rclpy.time.Time(seconds=0))
        except Exception as e:
            self.get_logger().error(f"TF error: {e}")
            return None

        p = PoseStamped()
        p.header.frame_id = gf
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = tf.transform.translation.x
        p.pose.position.y = tf.transform.translation.y
        p.pose.orientation.w = 1.0
        return p

    # ═══════════════════════════════════════════════════════
    #  Núcleo ARA*
    # ═══════════════════════════════════════════════════════

    def _heuristic(self, a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if self._heuristic_type == 'manhattan':
            return float(dx + dy)
        return math.hypot(dx, dy)

    def _f(self, g, h, eps):
        return g + eps * h

    def _improve_path(self, goal, eps, state, OPEN, CLOSED, INCONS):
        W, H = self._map.info.width, self._map.info.height

        if goal not in state:
            state[goal] = ARANode(goal[0], goal[1])

        while OPEN and state[goal].g > OPEN[0][0]:
            _, idx = heapq.heappop(OPEN)

            if idx in CLOSED:
                continue

            node = state[idx]
            node.v = node.g
            CLOSED.add(idx)

            for nx, ny, cost in self._neighbors(idx[0], idx[1], W, H):
                nidx = (nx, ny)
                if nidx not in state:
                    state[nidx] = ARANode(nx, ny)

                new_g = node.g + cost
                if state[nidx].g > new_g:
                    state[nidx].g = new_g
                    state[nidx].parent = idx

                    if nidx not in CLOSED:
                        h = self._heuristic(nidx, goal)
                        heapq.heappush(OPEN, (self._f(new_g, h, eps), nidx))
                    else:
                        INCONS.add(nidx)

    def _reconstruct(self, start, goal, state, header, x0, y0, res):
        path = Path()
        path.header = header

        cells = []
        cur = goal
        while cur is not None:
            cells.append(cur)
            if cur == start:
                break
            cur = state[cur].parent
        cells.reverse()

        last_yaw = 0.0
        for i, (ix, iy) in enumerate(cells):
            x, y = self._map_to_world(ix, iy, x0, y0, res)
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = x
            pose.pose.position.y = y

            if i + 1 < len(cells):
                nx, ny = self._map_to_world(
                    cells[i + 1][0], cells[i + 1][1], x0, y0, res)
                last_yaw = math.atan2(ny - y, nx - x)

            pose.pose.orientation = yaw_to_quaternion(last_yaw)
            path.poses.append(pose)

        return path

    def _plan(self, start: PoseStamped, goal: PoseStamped) -> Optional[Path]:
        info = self._map.info
        res, x0, y0 = info.resolution, info.origin.position.x, info.origin.position.y
        W, H = info.width, info.height

        s_idx = self._world_to_map(
            start.pose.position.x, start.pose.position.y,
            x0, y0, res, W, H)
        g_idx = self._world_to_map(
            goal.pose.position.x, goal.pose.position.y,
            x0, y0, res, W, H)

        if s_idx is None or g_idx is None:
            self.get_logger().error("Start o goal fuera del mapa")
            return None

        if self._log_svc:
            self.get_logger().info(
                f"Plan: start={s_idx} goal={g_idx}")

        # Snap a celda libre
        snap = self.get_parameter(
            'geometry.snap_start_goal_to_free').value
        if snap:
            max_r = max(1, int(math.ceil(
                float(self.get_parameter(
                    'geometry.snap_max_radius').value) / res)))

            if not self._is_free(s_idx):
                s_new = self._nearest_free(s_idx, max_r)
                if s_new is None:
                    self.get_logger().error(
                        f"Start {s_idx} bloqueado, sin libre cercana")
                    return None
                self.get_logger().warn(
                    f"Start snap: {s_idx} → {s_new}")
                s_idx = s_new

            if not self._is_free(g_idx):
                g_new = self._nearest_free(g_idx, max_r)
                if g_new is None:
                    self.get_logger().error(
                        f"Goal {g_idx} bloqueado, sin libre cercana")
                    return None
                self.get_logger().warn(
                    f"Goal snap: {g_idx} → {g_new}")
                g_idx = g_new
        else:
            if not self._is_free(s_idx):
                self.get_logger().error(f"Start {s_idx} bloqueado")
                return None
            if not self._is_free(g_idx):
                self.get_logger().error(f"Goal {g_idx} bloqueado")
                return None

        # Trivial
        if s_idx == g_idx:
            p = Path()
            p.header = start.header
            pose = PoseStamped()
            pose.header = start.header
            pose.pose.position.x, pose.pose.position.y = \
                self._map_to_world(s_idx[0], s_idx[1], x0, y0, res)
            pose.pose.orientation.w = 1.0
            p.poses.append(pose)
            return p

        # ARA* init
        eps = float(self.get_parameter('ara_core.epsilon_start').value)
        eps_dec = float(self.get_parameter('ara_core.epsilon_decrease').value)
        t_limit = float(self.get_parameter('ara_core.time_limit_sec').value)

        OPEN, CLOSED, INCONS = [], set(), set()
        state: Dict[Tuple[int, int], ARANode] = {}

        state[s_idx] = ARANode(s_idx[0], s_idx[1])
        state[s_idx].g = 0.0

        h0 = self._heuristic(s_idx, g_idx)
        heapq.heappush(OPEN, (self._f(0.0, h0, eps), s_idx))

        t0 = time.time()
        found = False

        debug_markers = MarkerArray()
        iter_n = 0

        # Bucle anytime
        while eps >= 1.0:
            self._improve_path(g_idx, eps, state, OPEN, CLOSED, INCONS)

            if g_idx in state and state[g_idx].g < float('inf'):
                found = True
                self.get_logger().info(
                    f"Ruta eps={eps:.2f}, "
                    f"g={state[g_idx].g:.1f}")

                if self._debug_mode:
                    cells = []
                    c = g_idx
                    while c is not None:
                        cells.append(c)
                        if c == s_idx:
                            break
                        c = state[c].parent
                    marker = self._debug_marker(
                        cells, eps, iter_n,
                        start.header.frame_id, x0, y0, res)
                    debug_markers.markers.append(marker)
                    iter_n += 1

            if (time.time() - t0) > t_limit:
                self.get_logger().warn("Tiempo agotado")
                break

            if eps == 1.0:
                break

            eps = max(1.0, eps - eps_dec)

            # Reconstruir OPEN con nuevo epsilon
            new_open = []
            for _, idx in OPEN:
                if idx not in CLOSED:
                    h = self._heuristic(idx, g_idx)
                    new_open.append(
                        (self._f(state[idx].g, h, eps), idx))
            for idx in INCONS:
                h = self._heuristic(idx, g_idx)
                new_open.append(
                    (self._f(state[idx].g, h, eps), idx))

            heapq.heapify(new_open)
            OPEN = new_open
            INCONS.clear()
            CLOSED.clear()

        # Debug markers
        if self._debug_mode and debug_markers.markers:
            dm = Marker()
            dm.action = Marker.DELETEALL
            debug_markers.markers.insert(0, dm)
            self.debug_pub.publish(debug_markers)

        if not found:
            self.get_logger().error(
                f"Sin ruta: start={s_idx} goal={g_idx} "
                f"estados={len(state)}")
            return None

        raw = self._reconstruct(
            s_idx, g_idx, state, start.header, x0, y0, res)

        # ══════════════════════════════════════════════════
        #  NUEVO: Simplificar con RDP antes de retornar
        # ══════════════════════════════════════════════════
        if self._rdp_tol > 0 and len(raw.poses) > 2:
            simplified = self._simplify_rdp(raw, self._rdp_tol)
            self.get_logger().info(
                f"RDP ({self._rdp_tol:.3f}m): "
                f"{len(raw.poses)} → {len(simplified.poses)} poses")
            return simplified

        return raw

    # ═══════════════════════════════════════════════════════
    #  NUEVO: Ramer-Douglas-Peucker
    # ═══════════════════════════════════════════════════════

    def _simplify_rdp(self, path: Path, tol: float) -> Path:
        """
        Simplifica un Path eliminando puntos redundantes.
        Mantiene la forma dentro de 'tol' metros de desviación.
        Típicamente reduce 10,000 → 200-500 puntos.
        """
        n = len(path.poses)
        if n <= 2:
            return path

        pts = [(p.pose.position.x, p.pose.position.y)
               for p in path.poses]

        keep = [False] * n
        keep[0] = True
        keep[-1] = True

        stack = [(0, n - 1)]

        while stack:
            i0, i1 = stack.pop()
            if i1 - i0 <= 1:
                continue

            sx, sy = pts[i0]
            ex, ey = pts[i1]
            dx, dy = ex - sx, ey - sy
            seg2 = dx * dx + dy * dy

            best_d, best_i = 0.0, i0

            for i in range(i0 + 1, i1):
                px, py = pts[i]
                if seg2 < 1e-12:
                    d = math.hypot(px - sx, py - sy)
                else:
                    t = max(0.0, min(1.0,
                        ((px - sx) * dx + (py - sy) * dy) / seg2))
                    d = math.hypot(px - (sx + t * dx),
                                   py - (sy + t * dy))
                if d > best_d:
                    best_d, best_i = d, i

            if best_d > tol:
                keep[best_i] = True
                stack.append((i0, best_i))
                stack.append((best_i, i1))

        out = Path()
        out.header = path.header
        out.poses = [path.poses[i] for i, k in enumerate(keep) if k]
        return out

    # ═══════════════════════════════════════════════════════
    #  Debug markers
    # ═══════════════════════════════════════════════════════

    def _debug_marker(self, cells, eps, mid, frame, x0, y0, res):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "ara_eps_paths"
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.01

        eps_start = self.get_parameter('ara_core.epsilon_start').value
        ratio = (eps - 1.0) / max(eps_start - 1.0, 0.01)

        m.color = ColorRGBA()
        m.color.r = max(0.0, min(1.0, float(ratio)))
        m.color.g = max(0.0, min(1.0, float(1.0 - ratio)))
        m.color.b = 0.0
        m.color.a = 0.8

        for ix, iy in cells:
            x, y = self._map_to_world(ix, iy, x0, y0, res)
            p = Point()
            p.x, p.y, p.z = x, y, mid * 0.02
            m.points.append(p)

        return m


def main(args=None):
    rclpy.init(args=args)
    node = ARAPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()