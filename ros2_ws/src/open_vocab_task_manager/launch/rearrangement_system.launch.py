#!/usr/bin/env python3
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _share(pkg: str, *parts: str) -> str:
    return os.path.join(get_package_share_directory(pkg), *parts)


def generate_launch_description() -> LaunchDescription:
    observer_cfg_default = _share("open_vocab_observer", "config", "observer.yaml")
    local_cfg_default = _share("open_vocab_local_perception", "config", "wrist_local_perception.yaml")
    grasp_cfg_default = _share("open_vocab_grasp_planner", "config", "grasp_planner.yaml")
    pick_cfg_default = _share("open_vocab_manipulation", "config", "manipulation.yaml")
    place_cfg_default = _share("open_vocab_manipulation", "config", "place.yaml")
    task_cfg_default = _share("open_vocab_task_manager", "config", "task_manager.yaml")
    home_cfg_default = _share("open_vocab_task_manager", "config", "home.yaml")
    # Historical filename kept for CLI compatibility; V10.5 treats it as generic
    # place-geometry policy + optional overrides, never as a semantic whitelist.
    profiles_default = _share("open_vocab_task_manager", "config", "destination_profiles.yaml")

    observer_config = LaunchConfiguration("observer_config")
    local_config = LaunchConfiguration("local_config")
    grasp_config = LaunchConfiguration("grasp_config")
    pick_config = LaunchConfiguration("pick_config")
    place_config = LaunchConfiguration("place_config")
    task_config = LaunchConfiguration("task_config")
    home_config = LaunchConfiguration("home_config")
    destination_profiles = LaunchConfiguration("destination_profiles")
    real_execution = ParameterValue(LaunchConfiguration("enable_real_execution"), value_type=bool)

    observer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_share("open_vocab_observer", "launch", "observation_stage.launch.py")),
        launch_arguments={
            "config": observer_config,
            "execute_motion": LaunchConfiguration("enable_real_execution"),
        }.items(),
    )

    # Launch local perception directly so TaskManager owns triggering and we can force
    # auto_trigger=false even if an older YAML was edited differently.
    local_node = Node(
        package="open_vocab_local_perception",
        executable="wrist_rgbd_localizer",
        name="open_vocab_wrist_rgbd_localizer",
        output="screen",
        parameters=[local_config, {"auto_trigger": False}],
    )

    grasp_node = Node(
        package="open_vocab_grasp_planner",
        executable="grasp_plan_node",
        name="open_vocab_grasp_plan",
        output="screen",
        parameters=[grasp_config],
    )

    pick_node = Node(
        package="open_vocab_manipulation",
        executable="pick_executor_node",
        name="open_vocab_pick_executor",
        output="screen",
        parameters=[
            pick_config,
            {
                "enable_arm_execution": real_execution,
                "enable_gripper_execution": real_execution,
                "enable_grasp_and_lift": real_execution,
                "allow_approach_dry_run": real_execution,
                "require_source_revalidation": False,
            },
        ],
    )

    place_node = Node(
        package="open_vocab_manipulation",
        executable="place_executor_node",
        name="open_vocab_place_executor",
        output="screen",
        parameters=[
            place_config,
            {
                "enable_place_arm_execution": real_execution,
                "enable_release_execution": real_execution,
            },
        ],
    )

    home_node = Node(
        package="open_vocab_task_manager",
        executable="home_executor_node",
        name="open_vocab_home_executor",
        output="screen",
        parameters=[
            home_config,
            {"enable_home_execution": real_execution},
        ],
    )

    task_node = Node(
        package="open_vocab_task_manager",
        executable="task_manager_node",
        name="open_vocab_task_manager",
        output="screen",
        parameters=[task_config, {"destination_profiles_path": destination_profiles}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("observer_config", default_value=observer_cfg_default),
        DeclareLaunchArgument("local_config", default_value=local_cfg_default),
        DeclareLaunchArgument("grasp_config", default_value=grasp_cfg_default),
        DeclareLaunchArgument("pick_config", default_value=pick_cfg_default),
        DeclareLaunchArgument("place_config", default_value=place_cfg_default),
        DeclareLaunchArgument("task_config", default_value=task_cfg_default),
        DeclareLaunchArgument("home_config", default_value=home_cfg_default),
        DeclareLaunchArgument("destination_profiles", default_value=profiles_default),
        DeclareLaunchArgument(
            "enable_real_execution",
            default_value="false",
            description=(
                "Master real-motion lock. Set true only after robot/table/gripper geometry and "
                "HOME and the existing V8 Pick/Place stack have been physically verified."
            ),
        ),
        observer_launch,
        local_node,
        grasp_node,
        pick_node,
        place_node,
        home_node,
        task_node,
    ])
