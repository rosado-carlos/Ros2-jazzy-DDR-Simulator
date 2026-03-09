import math
import random
import numpy as np
import heapq
import itertools

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy


class Vertex:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = float("inf")


class BITStar:

    def __init__(self, start, goal, grid, resolution, origin):

        self.start = Vertex(start[0], start[1])
        self.goal = Vertex(goal[0], goal[1])

        self.start.cost = 0.0

        self.grid = grid
        self.resolution = resolution
        self.origin = origin

        self.height, self.width = grid.shape

        self.V = {self.start}
        self.E = set()

        self.X_sample = {self.goal}

        self.Q_V = []
        self.Q_E = []

        self.batch_size = 400
        self.radius = 1.5

        self.counter = itertools.count()

    def planning(self):

        self.sample_batch()

        heapq.heappush(self.Q_V, (0.0, next(self.counter), self.start))

        for _ in range(5000):

            if not self.Q_E:

                if not self.Q_V:

                    self.sample_batch()

                    for v in self.V:
                        heapq.heappush(
                            self.Q_V,
                            (v.cost + self.heuristic(v),
                             next(self.counter),
                             v)
                        )

                else:

                    _, _, v = heapq.heappop(self.Q_V)
                    self.expand_vertex(v)

            else:

                _, _, v, x = heapq.heappop(self.Q_E)

                if self.collision(v, x):
                    continue

                new_cost = v.cost + self.dist(v, x)

                new = Vertex(x.x, x.y)
                new.parent = v
                new.cost = new_cost

                self.V.add(new)
                self.E.add((v, new))

                if x in self.X_sample:
                    self.X_sample.remove(x)

                if self.reached_goal(new):
                    return self.extract_path(new)

                heapq.heappush(
                    self.Q_V,
                    (new.cost + self.heuristic(new),
                     next(self.counter),
                     new)
                )

        return None

    def sample_batch(self):

        for _ in range(self.batch_size):

            x = random.uniform(
                self.origin[0],
                self.origin[0] + self.width * self.resolution
            )

            y = random.uniform(
                self.origin[1],
                self.origin[1] + self.height * self.resolution
            )

            gx, gy = self.world_to_grid(x, y)

            if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
                continue

            val = self.grid[gy, gx]

            if val == 100 or val == -1:
                continue

            self.X_sample.add(Vertex(x, y))

    def expand_vertex(self, v):

        for x in list(self.X_sample):

            if self.dist(v, x) > self.radius:
                continue

            cost = v.cost + self.dist(v, x) + self.heuristic(x)

            heapq.heappush(
                self.Q_E,
                (cost, next(self.counter), v, x)
            )

    def reached_goal(self, node):

        return self.dist(node, self.goal) < 0.5

    def heuristic(self, node):

        return math.hypot(
            node.x - self.goal.x,
            node.y - self.goal.y
        )

    def dist(self, n1, n2):

        return math.hypot(
            n1.x - n2.x,
            n1.y - n2.y
        )

    def extract_path(self, node):

        path = []

        while node:
            path.append((node.x, node.y))
            node = node.parent

        return path[::-1]

    def world_to_grid(self, x, y):

        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)

        return gx, gy

    def collision(self, n1, n2):

        d = self.dist(n1, n2)

        if d < 1e-6:
            return False

        steps = max(1, int(d / self.resolution))

        for i in range(steps + 1):

            t = i / steps

            x = n1.x + t * (n2.x - n1.x)
            y = n1.y + t * (n2.y - n1.y)

            gx, gy = self.world_to_grid(x, y)

            if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
                return True

            val = self.grid[gy, gx]

            if val == 100 or val == -1:
                return True

        return False


class BITStarNode(Node):

    def __init__(self):

        super().__init__("bitstar_planner")

        self.map = None
        self.map_info = None

        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            map_qos
        )

        self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_callback,
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            "/bitstar_path",
            map_qos
        )

        self.tree_pub = self.create_publisher(
            Marker,
            "/bitstar_tree",
            map_qos
        )

        self.get_logger().info("BIT* planner node ready")

    def map_callback(self, msg):

        self.map_info = msg.info

        self.map = np.array(msg.data, dtype=np.int16).reshape(
            msg.info.height,
            msg.info.width
        )

        self.get_logger().info(f"Free cells: {np.sum(self.map == 0)}")
        self.get_logger().info(f"Occupied cells: {np.sum(self.map == 100)}")
        self.get_logger().info(f"Unknown cells: {np.sum(self.map == -1)}")

    def goal_callback(self, msg):

        if self.map is None:
            self.get_logger().warn("No map yet")
            return

        start = (0.0, 0.0)

        goal = (
            msg.pose.position.x,
            msg.pose.position.y
        )

        planner = BITStar(
            start,
            goal,
            self.map,
            self.map_info.resolution,
            (
                self.map_info.origin.position.x,
                self.map_info.origin.position.y
            )
        )

        path = planner.planning()

        if path is None:
            self.get_logger().warn("No path found")
            return

        ros_path = Path()
        ros_path.header.frame_id = "map"

        for x, y in path:

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y

            ros_path.poses.append(pose)

        self.path_pub.publish(ros_path)

        self.publish_tree(planner)

        self.get_logger().info("Path found")

    def publish_tree(self, planner):

        marker = Marker()

        marker.header.frame_id = "map"
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.02

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for v1, v2 in planner.E:

            p1 = Point()
            p1.x = v1.x
            p1.y = v1.y

            p2 = Point()
            p2.x = v2.x
            p2.y = v2.y

            marker.points.append(p1)
            marker.points.append(p2)

        self.tree_pub.publish(marker)


def main():

    rclpy.init()

    node = BITStarNode()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()