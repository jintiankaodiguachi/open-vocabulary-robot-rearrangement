from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("open_vocab_grasp_planner")
    default_cfg = os.path.join(pkg_share, "config", "grasp_planner.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_cfg),
        Node(
            package="open_vocab_grasp_planner",
            executable="grasp_plan_node",
            name="open_vocab_grasp_plan",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),
    ])
