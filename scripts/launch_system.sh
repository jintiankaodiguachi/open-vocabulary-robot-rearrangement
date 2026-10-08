#!/usr/bin/env bash
# Configure and build first; ROS setup and external hardware bringup are prerequisites.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
configs="$repo_root/config/local/ros"
for name in observer wrist_local_perception grasp_planner manipulation place task_manager home destination_profiles; do
  if [[ ! -f "$configs/$name.yaml" ]]; then
    printf 'Missing %s; run python3 scripts/configure.py first.\n' "$configs/$name.yaml" >&2
    exit 1
  fi
done
exec ros2 launch open_vocab_task_manager rearrangement_system.launch.py \
  "observer_config:=$configs/observer.yaml" \
  "local_config:=$configs/wrist_local_perception.yaml" \
  "grasp_config:=$configs/grasp_planner.yaml" \
  "pick_config:=$configs/manipulation.yaml" \
  "place_config:=$configs/place.yaml" \
  "task_config:=$configs/task_manager.yaml" \
  "home_config:=$configs/home.yaml" \
  "destination_profiles:=$configs/destination_profiles.yaml" \
  "$@"
