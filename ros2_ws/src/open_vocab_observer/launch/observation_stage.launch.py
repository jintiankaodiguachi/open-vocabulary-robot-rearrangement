from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('open_vocab_observer'),
        'config',
        'observer.yaml',
    ])

    config_arg = DeclareLaunchArgument(
        'config',
        default_value=default_config,
        description='Path to observer.yaml',
    )
    execute_arg = DeclareLaunchArgument(
        'execute_motion',
        default_value='false',
        description='Allow real observation motion. Kept false unless a higher-level bringup explicitly enables it.',
    )

    config = LaunchConfiguration('config')
    execute_motion = ParameterValue(LaunchConfiguration('execute_motion'), value_type=bool)

    return LaunchDescription([
        config_arg,
        execute_arg,
        Node(
            package='open_vocab_observer',
            executable='scene_bridge.py',
            name='open_vocab_scene_bridge',
            output='screen',
            parameters=[config],
        ),
        Node(
            package='open_vocab_observer',
            executable='observation_pose_node.py',
            name='open_vocab_observation_pose',
            output='screen',
            parameters=[config],
        ),
        Node(
            package='open_vocab_observer',
            executable='moveit_observer',
            name='open_vocab_moveit_observer',
            output='screen',
            parameters=[config, {'execute_motion': execute_motion}],
        ),
    ])
