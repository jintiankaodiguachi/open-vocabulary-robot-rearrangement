cmake_minimum_required(VERSION 3.8)
project(open_vocab_observer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(open_vocab_interfaces REQUIRED)

add_executable(moveit_observer src/moveit_observer.cpp)
ament_target_dependencies(moveit_observer
  rclcpp
  rclcpp_action
  std_msgs
  geometry_msgs
  shape_msgs
  moveit_msgs
)
target_compile_features(moveit_observer PUBLIC cxx_std_17)

install(TARGETS
  moveit_observer
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/scene_bridge.py
  scripts/observation_pose_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
