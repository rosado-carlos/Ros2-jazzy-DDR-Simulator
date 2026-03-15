#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from tf2_ros import Buffer, TransformListener, TransformException

import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy


class MapWaypointMission(Node):

    def __init__(self):

        super().__init__('map_waypoint_mission')
        
        map_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(
            1.0,
            self.timer_callback
        )

        self.map = None
        self.waypoints = []
        self.current_wp = 0

        self.reach_threshold = 0.5
        self.step_cells = 50

        self.goal_active = False

        self.get_logger().info("Waypoint mission node ready")

    def map_callback(self, msg):

        if self.map is not None:
            return

        self.get_logger().info("Map recieved")

        self.map = msg

        self.generate_waypoints()

    def generate_waypoints(self):

        width = self.map.info.width
        height = self.map.info.height
        res = self.map.info.resolution

        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y

        step = self.step_cells

        for j in range(0, height, step):

            row_points = []

            for i in range(0, width, step):

                idx = j * width + i

                if self.map.data[idx] == 0:

                    x = origin_x + (i * res)
                    y = origin_y + (j * res)

                    row_points.append((x, y))

            if (j // step) % 2 == 0:
                self.waypoints.extend(row_points)
            else:
                self.waypoints.extend(reversed(row_points))

        self.get_logger().info(
            f"Generated {len(self.waypoints)} waypoints"
        )

    def timer_callback(self):

        if not self.waypoints:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )

        except TransformException as ex:

            self.get_logger().warn(f"TF error: {ex}")
            return

        rx = transform.transform.translation.x
        ry = transform.transform.translation.y

        if self.current_wp >= len(self.waypoints):

            self.get_logger().info("Mission completed")
            return

        wx, wy = self.waypoints[self.current_wp]

        dist = math.sqrt((wx - rx)**2 + (wy - ry)**2)

        if not self.goal_active:

            self.send_goal(wx, wy)
            self.goal_active = True
            return

        if dist < self.reach_threshold:

            self.get_logger().info(
                f"Waypoint {self.current_wp} reached"
            )

            self.current_wp += 1
            self.goal_active = False

    def send_goal(self, x, y):

        goal = PoseStamped()

        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)

        self.get_logger().info(
            f"Sending waypoint {self.current_wp}: {x:.2f}, {y:.2f}"
        )


def main(args=None):

    rclpy.init(args=args)

    node = MapWaypointMission()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()