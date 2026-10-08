import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("open_vocab_manipulation")
    config = LaunchConfiguration("config")
    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value=os.path.join(package_share, "config", "place.yaml"),
        ),
        Node(
            package="open_vocab_manipulation",
            executable="place_executor_node",
            name="open_vocab_place_executor",
            output="screen",
            parameters=[config],
        ),
    ])
