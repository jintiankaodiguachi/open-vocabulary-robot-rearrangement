#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        "config",
        default_value="",
        description="Absolute path to wrist_local_perception.yaml",
    )

    config = LaunchConfiguration("config")

    node = Node(
        package="open_vocab_local_perception",
        executable="wrist_rgbd_localizer",
        name="open_vocab_wrist_rgbd_localizer",
        output="screen",
        parameters=[config],
    )

    return LaunchDescription([
        config_arg,
        node,
    ])
