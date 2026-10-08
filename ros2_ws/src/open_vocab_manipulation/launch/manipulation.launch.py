import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("open_vocab_manipulation")
    pick_config = LaunchConfiguration("pick_config")
    place_config = LaunchConfiguration("place_config")
    return LaunchDescription([
        DeclareLaunchArgument(
            "pick_config",
            default_value=os.path.join(package_share, "config", "manipulation.yaml"),
        ),
        DeclareLaunchArgument(
            "place_config",
            default_value=os.path.join(package_share, "config", "place.yaml"),
        ),
        Node(
            package="open_vocab_manipulation",
            executable="pick_executor_node",
            name="open_vocab_pick_executor",
            output="screen",
            parameters=[pick_config],
        ),
        Node(
            package="open_vocab_manipulation",
            executable="place_executor_node",
            name="open_vocab_place_executor",
            output="screen",
            parameters=[place_config],
        ),
    ])
