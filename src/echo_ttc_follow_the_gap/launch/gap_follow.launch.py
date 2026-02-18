from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---------------- Launch Arguments ----------------
    ttc_min_arg = DeclareLaunchArgument(
        'ttc_min',
        default_value='0.6',
        description='Minimum TTC threshold'
    )

    v_max_arg = DeclareLaunchArgument(
        'v_max',
        default_value='2.5',
        description='Maximum forward velocity'
    )

    kp_steering_arg = DeclareLaunchArgument(
        'kp_steering',
        default_value='1.2',
        description='Steering proportional gain'
    )

    # ---------------- Configurations ----------------
    ttc_min = LaunchConfiguration('ttc_min')
    v_max = LaunchConfiguration('v_max')
    kp_steering = LaunchConfiguration('kp_steering')

    # ---------------- Nodes ----------------
    gap_finder_node = Node(
        package='echo_ttc_follow_the_gap',
        executable='ttc_gap_finder',
        name='ttc_gap_finder',
        output='screen',
        parameters=[
            {'ttc_min': ttc_min}
        ]
    )

    control_node = Node(
        package='echo_ttc_follow_the_gap',
        executable='ttc_control',
        name='ttc_control',
        output='screen',
        parameters=[
            {'ttc_min': ttc_min},
            {'v_max': v_max},
            {'kp_steering': kp_steering}
        ]
    )

    # ---------------- Launch Description ----------------
    return LaunchDescription([
        ttc_min_arg,
        v_max_arg,
        kp_steering_arg,
        gap_finder_node,
        control_node
    ])
