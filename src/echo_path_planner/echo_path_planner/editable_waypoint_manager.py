#!/usr/bin/env python3

import copy
import os
import threading

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
    MarkerArray,
)

from interactive_markers.interactive_marker_server import InteractiveMarkerServer


DEFAULT_WAYPOINTS = [
    {'id': 'G1', 'x': 1.488, 'y': -0.226, 'z': 0.0},
    {'id': 'G2', 'x': 3.55, 'y': -0.284, 'z': 0.0},
    {'id': 'G3', 'x': 5.654, 'y': -0.156, 'z': 0.0},
    {'id': 'G4', 'x': 7.649, 'y': -0.091, 'z': 0.0},
    {'id': 'G5', 'x': 9.396, 'y': -0.197, 'z': 0.0},
    {'id': 'G6', 'x': 11.621, 'y': -0.288, 'z': 0.0},
    {'id': 'G7', 'x': 13.54, 'y': -0.245, 'z': 0.0},
    {'id': 'G8', 'x': 15.099, 'y': 0.186, 'z': 0.0},
    {'id': 'G9', 'x': 15.885, 'y': 1.373, 'z': 0.0},
    {'id': 'G10', 'x': 15.954, 'y': 3.18, 'z': 0.0},
    {'id': 'G11', 'x': 16.101, 'y': 5.651, 'z': 0.0},
    {'id': 'G12', 'x': 15.801, 'y': 8.089, 'z': 0.0},
    {'id': 'G13', 'x': 15.935307502746582, 'y': 10.095335006713867, 'z': 0.0},
    {'id': 'G14', 'x': 16.007556915283203, 'y': 12.005492210388184, 'z': 0.0},
    {'id': 'G15', 'x': 14.741806983947754, 'y': 13.029191970825195, 'z': 0.0},
    {'id': 'G16', 'x': 12.491887092590332, 'y': 13.248990058898926, 'z': 0.0},
    {'id': 'G17', 'x': 10.126585006713867, 'y': 13.35079574584961, 'z': 0.0},
    {'id': 'G18', 'x': 7.563759803771973, 'y': 13.448494911193848, 'z': 0.0},
    {'id': 'G19', 'x': 5.0940093994140625, 'y': 13.718990325927734, 'z': 0.0},
    {'id': 'G20', 'x': 2.4488072395324707, 'y': 13.770964622497559, 'z': 0.0},
    {'id': 'G21', 'x': 0.14951463043689728, 'y': 13.677379608154297, 'z': 0.0},
    {'id': 'G22', 'x': -0.9055798053741455, 'y': 12.123811721801758, 'z': 0.0},
    {'id': 'G23', 'x': -0.949, 'y': 10.457, 'z': 0.0},
    {'id': 'G24', 'x': -0.975, 'y': 8.12, 'z': 0.0},
    {'id': 'G25', 'x': -0.937, 'y': 6.206, 'z': 0.0},
    {'id': 'G26', 'x': -0.11064048111438751, 'y': 4.223278999328613, 'z': 0.0},
    {'id': 'G27', 'x': 2.0908572673797607, 'y': 3.83951997756958, 'z': 0.0},
    {'id': 'G28', 'x': 4.445061683654785, 'y': 3.943472385406494, 'z': 0.0},
    {'id': 'G29', 'x': 6.650842666625977, 'y': 4.511137008666992, 'z': 0.0},
    {'id': 'G30', 'x': 8.39448356628418, 'y': 5.530646324157715, 'z': 0.0},
    {'id': 'G31', 'x': 9.834332466125488, 'y': 6.580469608306885, 'z': 0.0},
    {'id': 'G32', 'x': 11.584918022155762, 'y': 7.3931498527526855, 'z': 0.0},
    {'id': 'G33', 'x': 13.589219093322754, 'y': 6.994250297546387, 'z': 0.0},
    {'id': 'G34', 'x': 13.860456466674805, 'y': 4.707655906677246, 'z': 0.0},
    {'id': 'G35', 'x': 13.195345878601074, 'y': 2.503107786178589, 'z': 0.0},
    {'id': 'G36', 'x': 11.576109886169434, 'y': 1.27586829662323, 'z': 0.0},
    {'id': 'G37', 'x': 9.33996295928955, 'y': 1.0616912841796875, 'z': 0.0},
    {'id': 'G38', 'x': 7.2213239669799805, 'y': 0.9605523943901062, 'z': 0.0},
    {'id': 'G39', 'x': 5.274074077606201, 'y': 0.7062783241271973, 'z': 0.0},
    {'id': 'G40', 'x': 2.8287453651428223, 'y': 0.3619100749492645, 'z': 0.0},
    {'id': 'G41', 'x': 1.7034305334091187, 'y': 0.8069460988044739, 'z': 0.0},
    {'id': 'G42', 'x': 1.4463485479354858, 'y': 1.6818772554397583, 'z': 0.0},
    {'id': 'G43', 'x': 1.02906334400177, 'y': 2.741631269454956, 'z': 0.0},
    {'id': 'G44', 'x': -0.11640647798776627, 'y': 3.1303162574768066, 'z': 0.0},
    {'id': 'G45', 'x': -0.7923170328140259, 'y': 2.1104862689971924, 'z': 0.0},
    {'id': 'G46', 'x': -0.5958452224731445, 'y': 0.5432074069976807, 'z': 0.0},
    {'id': 'G47', 'x': 0.6989859938621521, 'y': -0.2766232490539551, 'z': 0.0},
]


class FixedWaypoints(Node):
    """
    Nodo paralelo al MissionManager.

    - Publica los waypoints como MarkerArray con esfera + texto G1...G47.
    - Publica tambien PoseArray y Path para que otros nodos puedan consumirlos.
    - Crea interactive markers para mover cada waypoint desde RViz.
    - Guarda automaticamente las posiciones editadas en un YAML.
    """

    def __init__(self):
        super().__init__('editable_waypoint_manager')

        self.declare_parameter('frames.global_frame', 'map')

        self.declare_parameter(
            'waypoints_file',
            self.resolve_waypoints_file('config/fixed_waypoints.yaml'),
        )

        self.declare_parameter('topics.markers', '/fixed_waypoints/markers')
        self.declare_parameter('topics.pose_array', '/fixed_waypoints/poses')
        self.declare_parameter('topics.path', '/fixed_waypoints/path')

        # Nombre del servidor de interactive markers. En RViz usa:
        # Add -> InteractiveMarkers -> Update Topic: /fixed_waypoints/update
        self.declare_parameter('interactive_marker_server', 'fixed_waypoints')

        self.declare_parameter('auto_save', True)
        self.declare_parameter('reset_to_defaults', False)

        self.declare_parameter('marker.sphere_scale', 0.35)
        self.declare_parameter('marker.text_scale', 0.35)
        self.declare_parameter('marker.text_z_offset', 0.55)

        self.global_frame = self.get_parameter(
            'frames.global_frame'
        ).get_parameter_value().string_value

        self.waypoints_file = self.resolve_waypoints_file(
            self.get_parameter('waypoints_file').get_parameter_value().string_value
        )

        markers_topic = self.get_parameter(
            'topics.markers'
        ).get_parameter_value().string_value

        pose_array_topic = self.get_parameter(
            'topics.pose_array'
        ).get_parameter_value().string_value

        path_topic = self.get_parameter(
            'topics.path'
        ).get_parameter_value().string_value

        server_name = self.get_parameter(
            'interactive_marker_server'
        ).get_parameter_value().string_value

        self.auto_save = self.get_parameter(
            'auto_save'
        ).get_parameter_value().bool_value

        reset_to_defaults = self.get_parameter(
            'reset_to_defaults'
        ).get_parameter_value().bool_value

        self.sphere_scale = self.get_parameter(
            'marker.sphere_scale'
        ).get_parameter_value().double_value

        self.text_scale = self.get_parameter(
            'marker.text_scale'
        ).get_parameter_value().double_value

        self.text_z_offset = self.get_parameter(
            'marker.text_z_offset'
        ).get_parameter_value().double_value

        viz_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.markers_pub = self.create_publisher(
            MarkerArray,
            markers_topic,
            viz_qos,
        )

        self.pose_array_pub = self.create_publisher(
            PoseArray,
            pose_array_topic,
            viz_qos,
        )

        self.path_pub = self.create_publisher(
            Path,
            path_topic,
            viz_qos,
        )

        self.lock = threading.Lock()

        if reset_to_defaults:
            self.waypoints = copy.deepcopy(DEFAULT_WAYPOINTS)
            self.save_waypoints()
            self.get_logger().warn(
                'reset_to_defaults=true: overwrote waypoint YAML with defaults.'
            )
        else:
            self.waypoints = self.load_waypoints()

        self.server = InteractiveMarkerServer(self, server_name)
        self.create_interactive_markers()

        self.publish_all()

        self.get_logger().info('FixedWaypoints ready')
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')
        self.get_logger().info(f'MarkerArray topic: {markers_topic}')
        self.get_logger().info(f'PoseArray topic: {pose_array_topic}')
        self.get_logger().info(f'Path topic: {path_topic}')
        self.get_logger().info(f'Interactive marker update topic: /{server_name}/update')
        self.get_logger().info(f'Waypoint YAML: {self.waypoints_file}')

    # ============================================================
    # YAML path resolver
    # ============================================================
    def resolve_waypoints_file(self, path):

        path = os.path.expanduser(os.path.expandvars(str(path)))

        if os.path.isabs(path):
            return path

        cwd_candidate = os.path.abspath(path)
        if os.path.exists(cwd_candidate):
            return cwd_candidate

        package_name = 'echo_path_planner'
        filename = os.path.basename(path)
        this_file = os.path.abspath(__file__)
        parts = this_file.split(os.sep)

        workspace_root = None
        for marker_name in ('build', 'install'):
            if marker_name in parts:
                idx = parts.index(marker_name)
                workspace_root = os.sep.join(parts[:idx]) or os.sep
                break

        candidates = []

        if workspace_root is not None:
            candidates.append(
                os.path.join(
                    workspace_root,
                    'src',
                    package_name,
                    'config',
                    filename,
                )
            )
            candidates.append(os.path.join(workspace_root, path))

        try:
            from ament_index_python.packages import get_package_share_directory
            package_share = get_package_share_directory(package_name)
            candidates.append(os.path.join(package_share, 'config', filename))
        except Exception:
            pass

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        if candidates:
            return candidates[0]

        return cwd_candidate

    # ============================================================
    # Load / save
    # ============================================================
    def load_waypoints(self):
        if not os.path.exists(self.waypoints_file):
            self.get_logger().warn(
                f'Waypoint file not found. Creating default file: {self.waypoints_file}'
            )
            waypoints = copy.deepcopy(DEFAULT_WAYPOINTS)
            self.waypoints = waypoints
            self.save_waypoints()
            return waypoints

        try:
            with open(self.waypoints_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            self.get_logger().error(
                f'Could not read {self.waypoints_file}: {exc}. Using defaults.'
            )
            return copy.deepcopy(DEFAULT_WAYPOINTS)

        raw_waypoints = data.get('waypoints', data)

        if not isinstance(raw_waypoints, list):
            self.get_logger().error(
                'Invalid waypoint YAML. Expected key "waypoints" with a list. '
                'Using defaults.'
            )
            return copy.deepcopy(DEFAULT_WAYPOINTS)

        waypoints = []

        for index, item in enumerate(raw_waypoints):
            try:
                waypoint_id = str(item.get('id', f'G{index + 1}'))
                x = float(item['x'])
                y = float(item['y'])
                z = float(item.get('z', 0.0))
            except Exception as exc:
                self.get_logger().warn(
                    f'Skipping invalid waypoint at index {index}: {exc}'
                )
                continue

            waypoints.append({'id': waypoint_id, 'x': x, 'y': y, 'z': z})

        if not waypoints:
            self.get_logger().warn('No valid waypoints found. Using defaults.')
            return copy.deepcopy(DEFAULT_WAYPOINTS)

        return waypoints

    def save_waypoints(self):
        directory = os.path.dirname(self.waypoints_file)

        if directory:
            os.makedirs(directory, exist_ok=True)

        data = {
            'frame_id': self.global_frame,
            'waypoints': copy.deepcopy(self.waypoints),
        }

        with open(self.waypoints_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

    # ============================================================
    # Interactive markers
    # ============================================================
    def create_interactive_markers(self):
        for waypoint in self.waypoints:
            int_marker = self.make_interactive_marker(waypoint)
            self.server.insert(
                int_marker,
                feedback_callback=self.process_feedback
            )

        self.server.applyChanges()

    def make_interactive_marker(self, waypoint):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.global_frame
        int_marker.name = waypoint['id']
        int_marker.description = waypoint['id']
        int_marker.scale = max(0.6, self.sphere_scale * 2.0)

        int_marker.pose.position.x = float(waypoint['x'])
        int_marker.pose.position.y = float(waypoint['y'])
        int_marker.pose.position.z = float(waypoint.get('z', 0.0))
        int_marker.pose.orientation.w = 1.0

        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.scale.x = self.sphere_scale
        sphere.scale.y = self.sphere_scale
        sphere.scale.z = self.sphere_scale
        sphere.color = self.make_color(0.1, 0.4, 1.0, 0.9)

        sphere_control = InteractiveMarkerControl()
        sphere_control.always_visible = True
        sphere_control.markers.append(sphere)
        int_marker.controls.append(sphere_control)

        move_control = InteractiveMarkerControl()
        move_control.name = 'move_xy'
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        move_control.orientation_mode = InteractiveMarkerControl.FIXED

        # Orienta el control para mover en el plano XY del frame map.
        move_control.orientation.w = 1.0
        move_control.orientation.x = 0.0
        move_control.orientation.y = 1.0
        move_control.orientation.z = 0.0

        int_marker.controls.append(move_control)

        return int_marker

    def process_feedback(self, feedback):
        if feedback.event_type not in (
            InteractiveMarkerFeedback.POSE_UPDATE,
            InteractiveMarkerFeedback.MOUSE_UP,
        ):
            return

        changed = False

        with self.lock:
            for waypoint in self.waypoints:
                if waypoint['id'] == feedback.marker_name:
                    waypoint['x'] = float(feedback.pose.position.x)
                    waypoint['y'] = float(feedback.pose.position.y)
                    waypoint['z'] = float(feedback.pose.position.z)
                    changed = True
                    break

            if not changed:
                return

            if self.auto_save and feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                self.save_waypoints()
                self.get_logger().info(
                    f'Saved {feedback.marker_name}: '
                    f'x={feedback.pose.position.x:.3f}, '
                    f'y={feedback.pose.position.y:.3f}'
                )

        self.publish_all()

    # ============================================================
    # Publishers
    # ============================================================
    def publish_all(self):
        with self.lock:
            waypoints_snapshot = copy.deepcopy(self.waypoints)

        self.publish_goal_markers(waypoints_snapshot)
        self.publish_pose_array(waypoints_snapshot)
        self.publish_path(waypoints_snapshot)

    def publish_pose_array(self, waypoints):
        msg = PoseArray()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        for waypoint in waypoints:
            pose = self.make_pose(waypoint)
            msg.poses.append(pose)

        self.pose_array_pub.publish(msg)

    def publish_path(self, waypoints):
        path = Path()
        path.header.frame_id = self.global_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for waypoint in waypoints:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose = self.make_pose(waypoint)
            path.poses.append(pose_stamped)

        self.path_pub.publish(path)

    def publish_goal_markers(self, waypoints):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.global_frame
        delete_marker.header.stamp = stamp
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, waypoint in enumerate(waypoints):
            pose = self.make_pose(waypoint)

            sphere = Marker()
            sphere.header.frame_id = self.global_frame
            sphere.header.stamp = stamp
            sphere.ns = 'fixed_waypoint_points'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = pose
            sphere.scale.x = self.sphere_scale
            sphere.scale.y = self.sphere_scale
            sphere.scale.z = self.sphere_scale
            sphere.color = self.make_color(0.1, 0.4, 1.0, 0.9)

            text = Marker()
            text.header.frame_id = self.global_frame
            text.header.stamp = stamp
            text.ns = 'fixed_waypoint_labels'
            text.id = 1000 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = pose.position.x
            text.pose.position.y = pose.position.y
            text.pose.position.z = pose.position.z + self.text_z_offset
            text.pose.orientation.w = 1.0
            text.scale.z = self.text_scale
            text.color = self.make_color(1.0, 1.0, 1.0, 1.0)
            text.text = waypoint['id']

            marker_array.markers.append(sphere)
            marker_array.markers.append(text)

        self.markers_pub.publish(marker_array)

    # ============================================================
    # Helpers
    # ============================================================
    def make_pose(self, waypoint):
        pose = Pose()
        pose.position.x = float(waypoint['x'])
        pose.position.y = float(waypoint['y'])
        pose.position.z = float(waypoint.get('z', 0.0))
        pose.orientation.w = 1.0
        return pose

    def make_color(self, r, g, b, a):
        color = ColorRGBA()
        color.r = float(r)
        color.g = float(g)
        color.b = float(b)
        color.a = float(a)
        return color


def main():
    rclpy.init()

    node = FixedWaypoints()

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