#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy


class MapWaypointMission(Node):

    def __init__(self):
        super().__init__("map_waypoint_mission")

        # parameters
        self.declare_parameter("margin", 1.0)
        self.declare_parameter("spacing", 10.0)

        self.margin = self.get_parameter("margin").value
        self.spacing = self.get_parameter("spacing").value

        self.waypoints = []
        self.index = 0
        self.map_loaded = False

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            map_qos
        )

        self.start_pub = self.create_publisher(
            PoseStamped,
            "/planner/start",
            map_qos
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            map_qos
        )

        self.get_logger().info("Waypoint mission node ready")

    # --------------------------------------------------

    def map_callback(self, msg):

        if self.map_loaded:
            return

        self.map_loaded = True

        width = msg.info.width * msg.info.resolution
        height = msg.info.height * msg.info.resolution

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        x_min = origin_x + self.margin
        x_max = origin_x + width - self.margin
        y_min = origin_y + self.margin
        y_max = origin_y + height - self.margin

        self.generate_rectangle_waypoints(x_min, x_max, y_min, y_max)

        self.get_logger().info(f"{len(self.waypoints)} waypoints generated")

        self.print_commands()

        self.console_loop()

    # --------------------------------------------------

    def generate_rectangle_waypoints(self, x_min, x_max, y_min, y_max):

        wp = []

        x = x_min
        while x <= x_max:
            wp.append((x, y_min))
            x += self.spacing

        y = y_min
        while y <= y_max:
            wp.append((x_max, y))
            y += self.spacing

        x = x_max
        while x >= x_min:
            wp.append((x, y_max))
            x -= self.spacing

        y = y_max
        while y >= y_min:
            wp.append((x_min, y))
            y -= self.spacing

        self.waypoints = wp

    # --------------------------------------------------

    def console_loop(self):

        while rclpy.ok():

            cmd = input("\ncommand > ").strip()

            if cmd == "n":
                self.publish_segment()
                self.index = (self.index + 1) % len(self.waypoints)

            elif cmd == "p":
                self.index = (self.index - 1) % len(self.waypoints)
                self.publish_segment()

            elif cmd == "r":
                self.index = 0
                print("Mission reset")

            elif cmd == "q":
                break

            elif cmd == "list":
                self.print_waypoints()

            else:
                print("Unknown command")

    # --------------------------------------------------

    def publish_segment(self):

        start_wp = self.waypoints[self.index]
        goal_wp = self.waypoints[(self.index + 1) % len(self.waypoints)]

        start = self.make_pose(start_wp[0], start_wp[1])
        goal = self.make_pose(goal_wp[0], goal_wp[1])

        self.start_pub.publish(start)
        self.goal_pub.publish(goal)

        self.get_logger().info(
            f"Segment {self.index} -> {(self.index+1)%len(self.waypoints)}"
            f" | start ({start_wp[0]:.2f},{start_wp[1]:.2f})"
            f" | goal ({goal_wp[0]:.2f},{goal_wp[1]:.2f})"
        )

    # --------------------------------------------------

    def make_pose(self, x, y):

        msg = PoseStamped()

        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0

        return msg

    # --------------------------------------------------

    def print_commands(self):

        print("\nWaypoint mission loaded\n")

        print("Commands:")
        print("n  → next waypoint segment")
        print("p  → previous segment")
        print("r  → reset mission")
        print("list → show waypoints")
        print("q  → quit\n")

    # --------------------------------------------------

    def print_waypoints(self):

        for i, wp in enumerate(self.waypoints):
            print(f"{i}: {wp}")


# --------------------------------------------------

def main():

    rclpy.init()

    node = MapWaypointMission()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()