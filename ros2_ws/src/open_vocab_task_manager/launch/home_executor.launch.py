#!/usr/bin/env python3
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("open_vocab_task_manager")
    config_default = os.path.join(share, "config", "home.yaml")
    config = LaunchConfiguration("config")
    execution = ParameterValue(LaunchConfiguration("enable_home_execution"), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=config_default),
        DeclareLaunchArgument(
            "enable_home_execution",
            default_value="false",
            description="Allow MoveIt execution to the persisted HOME joint configuration.",
        ),
        Node(
            package="open_vocab_task_manager",
            executable="home_executor_node",
            name="open_vocab_home_executor",
            output="screen",
            parameters=[config, {"enable_home_execution": execution}],
        ),
    ])
