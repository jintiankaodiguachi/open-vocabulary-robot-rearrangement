#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import rclpy
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from open_vocab_interfaces.action import Rearrange

from .journal import append_jsonl, atomic_write_json
from .profile_store import DestinationProfileStore, PROFILE_PARAMETER_KEYS
from .task_models import DestinationProfile, ServiceResult, TaskState


class TaskCancelled(RuntimeError):
    pass


class RearrangementTaskManager(Node):
    """Top-level action server that orchestrates the already-validated robot stack.

    The node intentionally does not reimplement perception, grasp planning, motion
    planning, or gripper control.  It composes the existing independently testable
    modules into one transactional task:

      HOME -> global source -> active observation -> local RGB-D -> grasp plan -> pick
      -> generic/override place geometry -> open-vocabulary global destination
      -> place -> optional verify -> HOME

    A single action goal is allowed at a time.  Cancellation sends best-effort
    software abort requests to both PickExecutor and PlaceExecutor.
    """

    STAGE_PROGRESS = {
        TaskState.PREFLIGHT: 0.01,
        TaskState.RESETTING: 0.03,
        TaskState.MOVE_HOME_START: 0.07,
        TaskState.GLOBAL_SOURCE: 0.12,
        TaskState.WAIT_OBSERVATION: 0.21,
        TaskState.LOCAL_SOURCE: 0.30,
        TaskState.PLAN_PICK: 0.39,
        TaskState.PREPARE_PICK: 0.46,
        TaskState.MOVE_PREGRASP: 0.54,
        TaskState.PICK_DRY_RUN: 0.60,
        TaskState.GRASP_AND_LIFT: 0.67,
        TaskState.CONFIGURE_DESTINATION: 0.71,
        TaskState.LOCATE_DESTINATION: 0.76,
        TaskState.PREPARE_PLACE: 0.82,
        TaskState.MOVE_PREPLACE: 0.88,
        TaskState.RELEASE_AND_RETREAT: 0.94,
        TaskState.VERIFY_PLACE: 0.97,
        TaskState.RETURN_HOME: 0.99,
        TaskState.DONE: 1.0,
        TaskState.CANCELLED: 1.0,
        TaskState.FAILED: 1.0,
        TaskState.IDLE: 0.0,
    }

    def __init__(self) -> None:
        super().__init__("open_vocab_task_manager")
        self.cb_group = ReentrantCallbackGroup()
        self._declare_parameters()
        self.p = self._load_parameters()

        self._goal_guard = threading.Lock()
        self._goal_reserved = False
        self._state_lock = threading.RLock()
        self._state = TaskState.IDLE
        self._active_task_id = ""
        self._active_source = ""
        self._active_destination = ""
        self._task_started_monotonic = 0.0

        self._observation_condition = threading.Condition()
        self._observation_done_seq = 0
        self._observation_last_monotonic = 0.0
        self.create_subscription(
            Bool,
            str(self.p["observation_done_topic"]),
            self._on_observation_done,
            10,
            callback_group=self.cb_group,
        )

        # V10.3: the historical open_vocab_observer does not publish an explicit
        # observation completion event.  Keep event support for future versions,
        # but add a deterministic joint-state settle fallback so the top-level
        # task does not depend on a topic that the current observer never creates.
        self._joint_condition = threading.Condition()
        self._latest_joint_positions: Dict[str, float] = {}
        self._latest_joint_velocities: Dict[str, float] = {}
        self._latest_joint_receive_monotonic = 0.0
        self.create_subscription(
            JointState,
            str(self.p["joint_states_topic"]),
            self._on_joint_state,
            qos_profile_sensor_data,
            callback_group=self.cb_group,
        )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(String, str(self.p["task_status_topic"]), qos)

        self.profile_store = DestinationProfileStore(str(self.p["destination_profiles_path"]))
        self._place_baseline: Optional[Dict[str, ParameterValue]] = None

        self._trigger_clients: Dict[str, Any] = {}
        for key in (
            "home_go_service",
            "home_abort_service",
            "home_status_service",
            "pick_reset_service",
            "place_reset_service",
            "local_perception_service",
            "grasp_plan_service",
            "pick_prepare_service",
            "pick_move_pregrasp_service",
            "pick_dry_run_service",
            "pick_grasp_lift_service",
            "pick_abort_service",
            "pick_status_service",
            "place_locate_service",
            "place_prepare_service",
            "place_move_preplace_service",
            "place_release_service",
            "place_verify_service",
            "place_abort_service",
            "place_status_service",
        ):
            name = str(self.p[key])
            self._trigger_clients[key] = self.create_client(
                Trigger, name, callback_group=self.cb_group
            )

        place_node = str(self.p["place_executor_node_name"]).rstrip("/")
        self._place_get_client = self.create_client(
            GetParameters,
            f"{place_node}/get_parameters",
            callback_group=self.cb_group,
        )
        self._place_set_client = self.create_client(
            SetParameters,
            f"{place_node}/set_parameters",
            callback_group=self.cb_group,
        )

        self._action_server = ActionServer(
            self,
            Rearrange,
            str(self.p["rearrange_action_name"]),
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.cb_group,
        )

        self._publish_status(TaskState.IDLE, 0.0, "TaskManager ready", {})
        self.get_logger().info(
            f"TaskManager ready: action={self.p['rearrange_action_name']}. "
            "One goal executes HOME -> source perception -> pick -> destination -> place -> HOME."
        )
        overrides = sorted(self.profile_store.names())
        self.get_logger().info(
            "Destination semantics are OPEN-VOCABULARY. Unknown destination names are "
            "forwarded directly to Grounded-SAM2 and use the generic place geometry. "
            + ("Geometry overrides: " + ", ".join(overrides) if overrides else "No geometry overrides configured.")
        )

    def _declare_parameters(self) -> None:
        d = self.declare_parameter
        # Action / status / audit
        d("rearrange_action_name", "/open_vocab/rearrange")
        d("task_status_topic", "/open_vocab/task/status")
        d("latest_task_json_path", "/opt/open-vocabulary-robot-rearrangement/perception/task_logs/latest_task.json")
        d("task_history_jsonl_path", "/opt/open-vocabulary-robot-rearrangement/perception/task_logs/task_history.jsonl")

        # Existing Stage-2A global source pipeline.  It remains the single source
        # of truth for latest_scene.json consumed by open_vocab_observer/scene_bridge.
        d("stage2a_root", "/opt/open-vocabulary-robot-rearrangement/perception/robot_host")
        d("stage2a_cli", "cli.py")
        d("stage2a_config", "config.yaml")
        d("stage2a_python", "python3")
        d("stage2a_pass_config_arg", False)
        d("stage2a_pass_json_flag", False)
        d("stage2a_scene_json", "outputs/latest_scene.json")
        d("stage2a_timeout_s", 30.0)
        d("min_source_confidence", 0.45)

        # Active-observation completion synchronization.
        # `auto` prefers an explicit Bool completion event when a publisher exists,
        # otherwise it falls back to joint-state motion-start + settle detection.
        # This keeps compatibility with the current validated observer, which moves
        # the robot correctly but does not publish /open_vocab/observation_motion_done.
        d("observation_done_topic", "/open_vocab/observation_motion_done")
        d("observation_completion_mode", "auto")  # auto | event | joint_settle
        d("observation_timeout_s", 60.0)
        d("observation_start_timeout_s", 25.0)
        d("observation_settle_time_s", 0.80)
        d("post_observation_settle_delay_s", 0.25)
        d("joint_states_topic", "/joint_states")
        d("observation_joint_names", [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ])
        d("observation_motion_start_delta_rad", 0.015)
        d("observation_velocity_start_rad_s", 0.020)
        d("observation_settle_position_delta_rad", 0.003)
        d("observation_settle_velocity_rad_s", 0.015)
        d("observation_joint_state_max_age_s", 1.0)
        d("required_observer_nodes", [
            "/open_vocab_scene_bridge",
            "/open_vocab_observation_pose",
            "/open_vocab_moveit_observer",
        ])
        # Deprecated V10/V10.2 parameter; retained only so old YAML files load.
        d("require_observation_publisher", False)

        # HOME/initial-point services. HOME is a deterministic joint-space pose.
        d("home_go_service", "/open_vocab/home/go")
        d("home_abort_service", "/open_vocab/home/abort")
        d("home_status_service", "/open_vocab/home/status")
        d("home_at_start", True)
        d("home_at_end", True)
        # Do not blindly move after an unknown manipulation failure by default.
        # A failed grasp may still hold an object or leave the robot near an obstacle.
        d("return_home_on_failure", False)

        # Trigger services
        d("pick_reset_service", "/open_vocab/pick/reset")
        d("place_reset_service", "/open_vocab/place/reset")
        d("local_perception_service", "/open_vocab/local_perception/trigger")
        d("grasp_plan_service", "/open_vocab/grasp/plan_latest")
        d("pick_prepare_service", "/open_vocab/pick/prepare")
        d("pick_move_pregrasp_service", "/open_vocab/pick/move_pregrasp")
        d("pick_dry_run_service", "/open_vocab/pick/approach_dry_run")
        d("pick_grasp_lift_service", "/open_vocab/pick/grasp_and_lift")
        d("pick_abort_service", "/open_vocab/pick/abort")
        d("pick_status_service", "/open_vocab/pick/status")
        d("place_locate_service", "/open_vocab/place/locate_destination")
        d("place_prepare_service", "/open_vocab/place/prepare")
        d("place_move_preplace_service", "/open_vocab/place/move_preplace")
        d("place_release_service", "/open_vocab/place/release_and_retreat")
        d("place_verify_service", "/open_vocab/place/verify")
        d("place_abort_service", "/open_vocab/place/abort")
        d("place_status_service", "/open_vocab/place/status")

        # Per-stage timeout.  Motion services block until their motion completes.
        d("service_discovery_timeout_s", 8.0)
        d("reset_timeout_s", 8.0)
        d("home_motion_timeout_s", 90.0)
        d("local_perception_timeout_s", 45.0)
        d("grasp_plan_timeout_s", 45.0)
        d("pick_prepare_timeout_s", 15.0)
        d("pick_motion_timeout_s", 90.0)
        d("place_locate_timeout_s", 35.0)
        d("place_prepare_timeout_s", 45.0)
        d("place_motion_timeout_s", 90.0)
        d("place_verify_timeout_s", 35.0)
        d("parameter_service_timeout_s", 8.0)

        # Destination profile -> dynamic PlaceExecutor parameters.
        d("destination_profiles_path", "")
        d("place_executor_node_name", "/open_vocab_place_executor")

        # Optional external service health checks. Empty items are ignored.
        d("require_external_health", False)
        d("external_health_urls", [
            "http://ai-server.local:8090/health",
            "http://ai-server.local:8091/health",
        ])
        d("external_health_timeout_s", 2.0)

    def _load_parameters(self) -> Dict[str, Any]:
        return {name: self.get_parameter(name).value for name in self._parameters.keys()}

    # ------------------------- ROS callbacks / action lifecycle -------------------------
    def _on_observation_done(self, msg: Bool) -> None:
        if not msg.data:
            return
        with self._observation_condition:
            self._observation_done_seq += 1
            self._observation_last_monotonic = time.monotonic()
            self._observation_condition.notify_all()

    def _on_joint_state(self, msg: JointState) -> None:
        positions: Dict[str, float] = {}
        velocities: Dict[str, float] = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                positions[str(name)] = float(msg.position[i])
            if i < len(msg.velocity):
                velocities[str(name)] = float(msg.velocity[i])
        if not positions:
            return
        with self._joint_condition:
            self._latest_joint_positions = positions
            self._latest_joint_velocities = velocities
            self._latest_joint_receive_monotonic = time.monotonic()
            self._joint_condition.notify_all()

    def _goal_callback(self, goal_request: Rearrange.Goal) -> GoalResponse:
        source = goal_request.source_query.strip()
        destination = goal_request.destination_query.strip()
        if not source or not destination:
            self.get_logger().warn("Rejecting rearrange goal: source/destination must be non-empty")
            return GoalResponse.REJECT
        with self._goal_guard:
            if self._goal_reserved:
                self.get_logger().warn("Rejecting rearrange goal: another task is active")
                return GoalResponse.REJECT
            self._goal_reserved = True
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().warn("Rearrange cancel requested; issuing best-effort Pick/Place aborts")
        self._request_abort_no_wait()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle) -> Rearrange.Result:
        task_id = str(uuid.uuid4())
        source = goal_handle.request.source_query.strip()
        destination = goal_handle.request.destination_query.strip()
        verify_place = bool(goal_handle.request.verify_place)
        use_pick_dry_run = bool(goal_handle.request.use_pick_dry_run)
        start = time.monotonic()
        failure_stage = ""
        result = Rearrange.Result()

        with self._state_lock:
            self._active_task_id = task_id
            self._active_source = source
            self._active_destination = destination
            self._task_started_monotonic = start

        try:
            self._stage(goal_handle, TaskState.PREFLIGHT, "Checking system readiness")
            self._preflight(goal_handle)

            self._stage(goal_handle, TaskState.RESETTING, "Resetting Pick/Place state machines")
            self._call_trigger("pick_reset_service", float(self.p["reset_timeout_s"]), goal_handle)
            self._call_trigger("place_reset_service", float(self.p["reset_timeout_s"]), goal_handle)

            if bool(self.p["home_at_start"]):
                self._stage(
                    goal_handle,
                    TaskState.MOVE_HOME_START,
                    "Moving robot to deterministic HOME/initial posture",
                )
                self._call_trigger(
                    "home_go_service", float(self.p["home_motion_timeout_s"]), goal_handle
                )

            with self._observation_condition:
                observation_seq_before = self._observation_done_seq
            observation_joint_baseline = self._snapshot_observation_joints(require_fresh=True)

            self._stage(goal_handle, TaskState.GLOBAL_SOURCE, f"Global open-vocabulary detection: {source}")
            source_scene = self._run_global_source(source, goal_handle)
            self._publish_status(
                TaskState.GLOBAL_SOURCE,
                self.STAGE_PROGRESS[TaskState.GLOBAL_SOURCE],
                f"Source localized: {source_scene.get('label', source)}",
                {
                    "source_scene": {
                        "label": source_scene.get("label"),
                        "confidence": source_scene.get("confidence"),
                        "coarse_base_xy": source_scene.get("coarse_base_xy"),
                        "debug_image_url": source_scene.get("debug_image_url"),
                    }
                },
            )

            self._stage(goal_handle, TaskState.WAIT_OBSERVATION, "Waiting for active wrist observation motion")
            self._wait_for_observation(
                observation_seq_before,
                observation_joint_baseline,
                goal_handle,
            )

            self._stage(goal_handle, TaskState.LOCAL_SOURCE, "Running wrist RGB-D local perception")
            self._call_trigger(
                "local_perception_service", float(self.p["local_perception_timeout_s"]), goal_handle
            )

            self._stage(goal_handle, TaskState.PLAN_PICK, "Running GraspNet + robot-aware grasp planning")
            self._call_trigger("grasp_plan_service", float(self.p["grasp_plan_timeout_s"]), goal_handle)

            self._stage(goal_handle, TaskState.PREPARE_PICK, "Applying pick execution safety gate")
            self._call_trigger("pick_prepare_service", float(self.p["pick_prepare_timeout_s"]), goal_handle)

            self._stage(goal_handle, TaskState.MOVE_PREGRASP, "Moving to pregrasp")
            self._call_trigger("pick_move_pregrasp_service", float(self.p["pick_motion_timeout_s"]), goal_handle)

            if use_pick_dry_run:
                self._stage(goal_handle, TaskState.PICK_DRY_RUN, "Executing verified Cartesian dry approach")
                self._call_trigger("pick_dry_run_service", float(self.p["pick_motion_timeout_s"]), goal_handle)

            self._stage(goal_handle, TaskState.GRASP_AND_LIFT, "Closing gripper and lifting source object")
            pick_result = self._call_trigger(
                "pick_grasp_lift_service", float(self.p["pick_motion_timeout_s"]), goal_handle
            )

            self._stage(
                goal_handle,
                TaskState.CONFIGURE_DESTINATION,
                f"Configuring place geometry for open-vocabulary destination: {destination}",
            )
            profile = self.profile_store.resolve_or_default(destination)
            applied = self._apply_destination_profile(profile, source, goal_handle)
            geometry_policy = "override" if profile.matched_override else "generic-startup-baseline"
            self._publish_status(
                TaskState.CONFIGURE_DESTINATION,
                self.STAGE_PROGRESS[TaskState.CONFIGURE_DESTINATION],
                f"Destination ready: query='{profile.query}', geometry={geometry_policy}",
                {"destination_geometry": applied},
            )

            # Semantic grounding stays open-vocabulary: for an unseen destination,
            # profile.query is exactly the raw user/VLM destination text.
            self._stage(goal_handle, TaskState.LOCATE_DESTINATION, f"Locating destination: {profile.query}")
            locate_result = self._call_trigger(
                "place_locate_service", float(self.p["place_locate_timeout_s"]), goal_handle
            )

            self._stage(goal_handle, TaskState.PREPARE_PLACE, "Searching robot-feasible place pose")
            prepare_place = self._call_trigger(
                "place_prepare_service", float(self.p["place_prepare_timeout_s"]), goal_handle
            )

            self._stage(goal_handle, TaskState.MOVE_PREPLACE, "Moving held object above destination")
            self._call_trigger(
                "place_move_preplace_service", float(self.p["place_motion_timeout_s"]), goal_handle
            )

            self._stage(goal_handle, TaskState.RELEASE_AND_RETREAT, "Releasing object and retreating")
            release_result = self._call_trigger(
                "place_release_service", float(self.p["place_motion_timeout_s"]), goal_handle
            )

            verify_message = ""
            if verify_place:
                self._stage(goal_handle, TaskState.VERIFY_PLACE, "Running coarse post-place verification")
                verify_result = self._call_trigger(
                    "place_verify_service", float(self.p["place_verify_timeout_s"]), goal_handle
                )
                verify_message = verify_result.message

            if bool(self.p["home_at_end"]):
                self._stage(
                    goal_handle,
                    TaskState.RETURN_HOME,
                    "Returning robot to HOME after successful placement",
                )
                self._call_trigger(
                    "home_go_service", float(self.p["home_motion_timeout_s"]), goal_handle
                )

            elapsed = time.monotonic() - start
            self._stage(goal_handle, TaskState.DONE, "Rearrangement completed", force_progress=1.0)
            goal_handle.succeed()
            result.success = True
            result.task_id = task_id
            result.final_state = TaskState.DONE.value
            result.failure_stage = ""
            result.message = (
                f"DONE: {source} -> {destination}. "
                f"pick=[{pick_result.message}] place=[{release_result.message}]"
                + (f" verify=[{verify_message}]" if verify_message else "")
            )
            result.total_time_s = float(elapsed)
            self._record_terminal(result, source, destination)
            return result

        except TaskCancelled as exc:
            elapsed = time.monotonic() - start
            self._set_state_only(TaskState.CANCELLED)
            goal_handle.canceled()
            result.success = False
            result.task_id = task_id
            result.final_state = TaskState.CANCELLED.value
            result.failure_stage = failure_stage or self._state.value
            result.message = str(exc)
            result.total_time_s = float(elapsed)
            self._publish_status(TaskState.CANCELLED, 1.0, str(exc), {})
            self._record_terminal(result, source, destination)
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            failure_stage = self._state.value
            self._request_abort_no_wait()
            # Default is false. Returning HOME after an unknown manipulation failure can be
            # unsafe if the gripper still holds an object or the robot is near an obstacle.
            if bool(self.p["return_home_on_failure"]):
                try:
                    self._call_trigger_no_cancel(
                        "home_go_service", float(self.p["home_motion_timeout_s"])
                    )
                except Exception as home_exc:
                    self.get_logger().error(f"Best-effort failure HOME recovery failed: {home_exc}")
            self._set_state_only(TaskState.FAILED)
            self.get_logger().error(f"Task {task_id} failed at {failure_stage}: {exc}\n{traceback.format_exc()}")
            goal_handle.abort()
            result.success = False
            result.task_id = task_id
            result.final_state = TaskState.FAILED.value
            result.failure_stage = failure_stage
            result.message = f"{type(exc).__name__}: {exc}"
            result.total_time_s = float(elapsed)
            self._publish_status(TaskState.FAILED, 1.0, result.message, {"failure_stage": failure_stage})
            self._record_terminal(result, source, destination)
            return result
        finally:
            with self._goal_guard:
                self._goal_reserved = False

    # ------------------------- stage helpers -------------------------
    def _stage(
        self,
        goal_handle,
        state: TaskState,
        message: str,
        *,
        force_progress: Optional[float] = None,
    ) -> None:
        self._check_cancel(goal_handle)
        progress = self.STAGE_PROGRESS[state] if force_progress is None else float(force_progress)
        self._set_state_only(state)
        feedback = Rearrange.Feedback()
        feedback.task_id = self._active_task_id
        feedback.state = state.value
        feedback.progress = float(progress)
        feedback.message = message
        feedback.elapsed_s = float(self._elapsed())
        goal_handle.publish_feedback(feedback)
        self._publish_status(state, progress, message, {})
        self.get_logger().info(f"[{self._active_task_id[:8]}] {state.value}: {message}")

    def _set_state_only(self, state: TaskState) -> None:
        with self._state_lock:
            self._state = state

    def _elapsed(self) -> float:
        if self._task_started_monotonic <= 0.0:
            return 0.0
        return time.monotonic() - self._task_started_monotonic

    def _publish_status(
        self,
        state: TaskState,
        progress: float,
        message: str,
        details: Dict[str, Any],
    ) -> None:
        payload = {
            "task_id": self._active_task_id or None,
            "state": state.value,
            "progress": float(progress),
            "message": message,
            "source_query": self._active_source or None,
            "destination_query": self._active_destination or None,
            "elapsed_s": float(self._elapsed()),
            "timestamp_unix_s": time.time(),
            "details": details,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        try:
            atomic_write_json(str(self.p["latest_task_json_path"]), payload)
        except Exception as exc:
            self.get_logger().error(f"Unable to write latest task status: {exc}")

    def _record_terminal(self, result: Rearrange.Result, source: str, destination: str) -> None:
        payload = {
            "task_id": result.task_id,
            "success": bool(result.success),
            "final_state": result.final_state,
            "failure_stage": result.failure_stage,
            "message": result.message,
            "source_query": source,
            "destination_query": destination,
            "total_time_s": float(result.total_time_s),
            "timestamp_unix_s": time.time(),
        }
        try:
            atomic_write_json(str(self.p["latest_task_json_path"]), payload)
            append_jsonl(str(self.p["task_history_jsonl_path"]), payload)
        except Exception as exc:
            self.get_logger().error(f"Unable to write task history: {exc}")

    # ------------------------- preflight -------------------------
    def _preflight(self, goal_handle) -> None:
        self._check_cancel(goal_handle)
        root = Path(str(self.p["stage2a_root"])).expanduser().resolve()
        cli = root / str(self.p["stage2a_cli"])
        cfg = root / str(self.p["stage2a_config"])
        if not root.is_dir() or not cli.is_file() or not cfg.is_file():
            raise RuntimeError(f"Stage2A paths invalid: root={root}, cli={cli}, config={cfg}")

        timeout = float(self.p["service_discovery_timeout_s"])
        for key, client in self._trigger_clients.items():
            if key in ("pick_abort_service", "place_abort_service"):
                # Abort availability is still checked, but we keep the same concise error path.
                pass
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"Required service unavailable: {self.p[key]}")

        if not self._place_get_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("PlaceExecutor get_parameters service unavailable")
        if not self._place_set_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("PlaceExecutor set_parameters service unavailable")

        mode = str(self.p["observation_completion_mode"]).strip().lower()
        if mode not in ("auto", "event", "joint_settle"):
            raise RuntimeError(
                "observation_completion_mode must be one of: auto, event, joint_settle"
            )

        # Check the actual observer pipeline nodes, not a completion topic that older
        # validated observer builds never published. Graph discovery can lag briefly.
        required_nodes = [str(x) for x in self.p["required_observer_nodes"]]
        deadline = time.monotonic() + timeout
        missing_nodes = required_nodes
        while time.monotonic() < deadline:
            self._check_cancel(goal_handle)
            present = self._graph_node_fqns()
            missing_nodes = [name for name in required_nodes if name not in present]
            if not missing_nodes:
                break
            time.sleep(0.05)
        if missing_nodes:
            raise RuntimeError(
                "Observer pipeline node(s) missing: " + ", ".join(missing_nodes)
            )

        event_publishers = self.count_publishers(str(self.p["observation_done_topic"]))
        if mode == "event" and event_publishers <= 0:
            raise RuntimeError(
                f"observation_completion_mode=event but no publisher exists on "
                f"{self.p['observation_done_topic']}"
            )
        if mode == "auto" and event_publishers <= 0:
            self.get_logger().warn(
                "No explicit observation completion publisher is present; "
                "using joint-state motion/settle synchronization for this task."
            )

        # Joint-state synchronization is required for joint_settle and for auto fallback.
        if mode == "joint_settle" or (mode == "auto" and event_publishers <= 0):
            self._snapshot_observation_joints(require_fresh=True)

        home_status = self._call_trigger("home_status_service", 5.0, goal_handle)
        pick_status = self._call_trigger("pick_status_service", 5.0, goal_handle)
        place_status = self._call_trigger("place_status_service", 5.0, goal_handle)
        try:
            home = json.loads(home_status.message)
            pick = json.loads(pick_status.message)
            place = json.loads(place_status.message)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Executor status is not JSON: {exc}") from exc

        if not bool(home.get("configured")):
            raise RuntimeError(
                "HOME/initial posture is not configured. Put the robot in the desired safe initial "
                "posture and call /open_vocab/home/capture_current once."
            )
        if (bool(self.p["home_at_start"]) or bool(self.p["home_at_end"])) and not bool(
            home.get("execution_enabled")
        ):
            raise RuntimeError(
                "HomeExecutor execution is locked. Launch rearrangement_system.launch.py with "
                "enable_real_execution:=true after HOME has been captured and tested."
            )

        required_pick = (
            bool(pick.get("arm_execution_enabled")),
            bool(pick.get("gripper_execution_enabled")),
            bool(pick.get("grasp_and_lift_enabled")),
        )
        if not all(required_pick):
            raise RuntimeError(
                "PickExecutor real execution is locked. Launch rearrangement_system.launch.py "
                "with enable_real_execution:=true after safety verification."
            )
        if not bool(place.get("place_arm_execution_enabled")) or not bool(place.get("release_execution_enabled")):
            raise RuntimeError(
                "PlaceExecutor real execution/release is locked. Launch with enable_real_execution:=true."
            )

        if self._place_baseline is None:
            self._place_baseline = self._get_place_parameters(goal_handle)

        if bool(self.p["require_external_health"]):
            timeout_http = float(self.p["external_health_timeout_s"])
            for url in self.p["external_health_urls"]:
                url = str(url).strip()
                if not url:
                    continue
                try:
                    resp = requests.get(url, timeout=timeout_http)
                    resp.raise_for_status()
                except Exception as exc:
                    raise RuntimeError(f"External health check failed: {url}: {exc}") from exc

    # ------------------------- global source + observation -------------------------
    def _run_global_source(self, source: str, goal_handle) -> Dict[str, Any]:
        root = Path(str(self.p["stage2a_root"])).expanduser().resolve()
        cli = root / str(self.p["stage2a_cli"])
        cfg = root / str(self.p["stage2a_config"])
        scene = root / str(self.p["stage2a_scene_json"])
        before_mtime = scene.stat().st_mtime_ns if scene.is_file() else None
        # Keep the one-key path behaviorally identical to the user's already
        # validated manual command: `python3 cli.py <query>`.  Optional flags are
        # explicit parameters rather than assumptions about cli.py's interface.
        cmd = [str(self.p["stage2a_python"]), str(cli), source]
        if bool(self.p["stage2a_pass_config_arg"]):
            cmd.extend(["--config", str(cfg)])
        if bool(self.p["stage2a_pass_json_flag"]):
            cmd.append("--json")
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        deadline = time.monotonic() + float(self.p["stage2a_timeout_s"])
        while proc.poll() is None:
            self._check_cancel(goal_handle, terminate_process=proc)
            if time.monotonic() > deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise TimeoutError(f"Stage2A global source perception timed out: {source}")
            time.sleep(0.05)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Stage2A failed rc={proc.returncode}: {stderr.strip() or stdout.strip()}"
            )
        if not scene.is_file():
            raise RuntimeError(f"Stage2A succeeded but latest_scene.json is missing: {scene}")
        stat = scene.stat()
        if before_mtime is not None and stat.st_mtime_ns == before_mtime:
            raise RuntimeError("Stage2A returned success but latest_scene.json was not updated")
        payload = json.loads(scene.read_text(encoding="utf-8"))

        # scene_bridge.py in the validated observer consumes an object store with
        # `objects: [...]`.  Older Stage2A snapshots may instead be a single object.
        # Accept both contracts and select the newest matching object exactly the
        # same way as scene_bridge.
        objects = payload.get("objects") if isinstance(payload, dict) else None
        if isinstance(objects, list) and objects:
            valid = [x for x in objects if isinstance(x, dict)]
            exact = [
                x for x in valid
                if str(x.get("query", "")).casefold() == source.casefold()
            ]
            pool = exact or valid
            if not pool:
                raise RuntimeError("Stage2A latest_scene.json contains no valid objects")
            data = max(pool, key=lambda x: float(x.get("timestamp", 0.0)))
        elif isinstance(payload, dict):
            data = payload
        else:
            raise RuntimeError("Stage2A latest_scene.json has unsupported JSON structure")

        scene_query = str(data.get("query", "")).strip()
        if scene_query and scene_query.casefold() != source.casefold():
            raise RuntimeError(
                f"Stage2A latest scene query mismatch: expected '{source}', got '{scene_query}'"
            )
        confidence = float(data.get("confidence", 0.0))
        if confidence < float(self.p["min_source_confidence"]):
            raise RuntimeError(
                f"Source confidence too low: {confidence:.3f} < {float(self.p['min_source_confidence']):.3f}"
            )
        return data

    def _graph_node_fqns(self) -> set[str]:
        fqns: set[str] = set()
        for name, namespace in self.get_node_names_and_namespaces():
            ns = str(namespace or "/")
            if ns == "/":
                fqns.add(f"/{name}")
            else:
                fqns.add(f"{ns.rstrip('/')}/{name}")
        return fqns

    def _snapshot_observation_joints(self, *, require_fresh: bool) -> Dict[str, float]:
        names = [str(x) for x in self.p["observation_joint_names"]]
        max_age = float(self.p["observation_joint_state_max_age_s"])
        deadline = time.monotonic() + max(2.0, float(self.p["service_discovery_timeout_s"]))
        while True:
            with self._joint_condition:
                positions = dict(self._latest_joint_positions)
                received = float(self._latest_joint_receive_monotonic)
            now = time.monotonic()
            missing = [name for name in names if name not in positions]
            age = now - received if received > 0.0 else float("inf")
            if not missing and (not require_fresh or age <= max_age):
                return {name: float(positions[name]) for name in names}
            if now >= deadline:
                if missing:
                    raise RuntimeError(
                        "Cannot synchronize observation motion: /joint_states is missing "
                        + ", ".join(missing)
                    )
                raise RuntimeError(
                    f"Cannot synchronize observation motion: /joint_states is stale "
                    f"(age={age:.2f}s > {max_age:.2f}s)"
                )
            with self._joint_condition:
                self._joint_condition.wait(timeout=0.10)

    def _wait_for_observation_event(self, seq_before: int, goal_handle, deadline: float) -> None:
        with self._observation_condition:
            while self._observation_done_seq <= seq_before:
                self._check_cancel(goal_handle)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Timed out waiting for explicit active-observation completion event."
                    )
                self._observation_condition.wait(timeout=min(0.20, remaining))

    def _wait_for_observation_joint_settle(
        self,
        baseline: Dict[str, float],
        goal_handle,
        deadline: float,
    ) -> None:
        names = list(baseline.keys())
        start_delta = float(self.p["observation_motion_start_delta_rad"])
        start_vel = float(self.p["observation_velocity_start_rad_s"])
        settle_delta = float(self.p["observation_settle_position_delta_rad"])
        settle_vel = float(self.p["observation_settle_velocity_rad_s"])
        settle_time = float(self.p["observation_settle_time_s"])
        start_deadline = min(
            deadline,
            time.monotonic() + float(self.p["observation_start_timeout_s"]),
        )
        max_age = float(self.p["observation_joint_state_max_age_s"])

        motion_started = False
        stable_since: Optional[float] = None
        previous: Optional[Dict[str, float]] = None
        last_max_from_baseline = 0.0

        while time.monotonic() < deadline:
            self._check_cancel(goal_handle)
            with self._joint_condition:
                positions_all = dict(self._latest_joint_positions)
                velocities_all = dict(self._latest_joint_velocities)
                received = float(self._latest_joint_receive_monotonic)

            now = time.monotonic()
            if received <= 0.0 or now - received > max_age:
                with self._joint_condition:
                    self._joint_condition.wait(timeout=0.05)
                continue
            if any(name not in positions_all for name in names):
                with self._joint_condition:
                    self._joint_condition.wait(timeout=0.05)
                continue

            current = {name: float(positions_all[name]) for name in names}
            max_from_baseline = max(
                abs(current[name] - baseline[name]) for name in names
            )
            last_max_from_baseline = max_from_baseline
            max_velocity = max(
                (abs(float(velocities_all.get(name, 0.0))) for name in names),
                default=0.0,
            )

            if not motion_started:
                if max_from_baseline >= start_delta or max_velocity >= start_vel:
                    motion_started = True
                    stable_since = None
                    previous = current
                    self.get_logger().info(
                        "Active-observation motion detected from /joint_states: "
                        f"max_delta={max_from_baseline:.4f}rad, "
                        f"max_velocity={max_velocity:.4f}rad/s"
                    )
                elif now >= start_deadline:
                    raise TimeoutError(
                        "Observer pipeline is running but no robot motion was detected after "
                        f"global source update (max joint delta={max_from_baseline:.4f}rad). "
                        "Check scene_bridge -> observation_pose -> moveit_observer execution."
                    )
            else:
                max_step = 0.0 if previous is None else max(
                    abs(current[name] - previous[name]) for name in names
                )
                settled = max_step <= settle_delta and max_velocity <= settle_vel
                if settled:
                    if stable_since is None:
                        stable_since = now
                    elif now - stable_since >= settle_time:
                        self.get_logger().info(
                            "Active-observation motion settled: "
                            f"max_from_start={max_from_baseline:.4f}rad, "
                            f"last_step={max_step:.4f}rad, "
                            f"max_velocity={max_velocity:.4f}rad/s"
                        )
                        delay = max(0.0, float(self.p["post_observation_settle_delay_s"]))
                        if delay > 0.0:
                            end_delay = time.monotonic() + delay
                            while time.monotonic() < end_delay:
                                self._check_cancel(goal_handle)
                                time.sleep(min(0.05, end_delay - time.monotonic()))
                        return
                else:
                    stable_since = None
                previous = current

            with self._joint_condition:
                self._joint_condition.wait(timeout=0.05)

        raise TimeoutError(
            "Timed out waiting for active observation motion to settle "
            f"(motion_started={motion_started}, "
            f"max_joint_delta={last_max_from_baseline:.4f}rad)."
        )

    def _wait_for_observation(
        self,
        seq_before: int,
        joint_baseline: Dict[str, float],
        goal_handle,
    ) -> None:
        deadline = time.monotonic() + float(self.p["observation_timeout_s"])
        mode = str(self.p["observation_completion_mode"]).strip().lower()
        publisher_present = self.count_publishers(str(self.p["observation_done_topic"])) > 0

        if mode == "event" or (mode == "auto" and publisher_present):
            self.get_logger().info(
                f"Observation completion sync: event topic {self.p['observation_done_topic']}"
            )
            self._wait_for_observation_event(seq_before, goal_handle, deadline)
            return

        self.get_logger().info(
            "Observation completion sync: /joint_states motion-start + settle fallback"
        )
        self._wait_for_observation_joint_settle(joint_baseline, goal_handle, deadline)

    # ------------------------- service / parameter transport -------------------------
    def _call_trigger(self, key: str, timeout_s: float, goal_handle) -> ServiceResult:
        client = self._trigger_clients[key]
        if not client.wait_for_service(timeout_sec=min(timeout_s, float(self.p["service_discovery_timeout_s"]))):
            raise RuntimeError(f"Service unavailable: {self.p[key]}")
        future = client.call_async(Trigger.Request())
        response = self._wait_future(future, timeout_s, goal_handle, str(self.p[key]))
        if response is None:
            raise RuntimeError(f"Service returned no response: {self.p[key]}")
        if not bool(response.success):
            raise RuntimeError(f"{self.p[key]} failed: {response.message}")
        return ServiceResult(True, str(response.message))

    def _call_trigger_no_cancel(self, key: str, timeout_s: float) -> ServiceResult:
        """Best-effort helper used only for optional failure recovery.

        It deliberately ignores Action cancellation state because it runs from exception
        cleanup, not the nominal transactional path.
        """
        client = self._trigger_clients[key]
        if not client.wait_for_service(timeout_sec=min(timeout_s, float(self.p["service_discovery_timeout_s"]))):
            raise RuntimeError(f"Service unavailable: {self.p[key]}")
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for {self.p[key]} after {timeout_s:.1f}s")
            time.sleep(0.02)
        exc = future.exception()
        if exc is not None:
            raise RuntimeError(f"{self.p[key]} transport error: {exc}") from exc
        response = future.result()
        if response is None or not bool(response.success):
            message = "no response" if response is None else str(response.message)
            raise RuntimeError(f"{self.p[key]} failed: {message}")
        return ServiceResult(True, str(response.message))

    def _wait_future(self, future, timeout_s: float, goal_handle, label: str):
        deadline = time.monotonic() + timeout_s
        while not future.done():
            self._check_cancel(goal_handle)
            if time.monotonic() > deadline:
                self._request_abort_no_wait()
                raise TimeoutError(f"Timed out waiting for {label} after {timeout_s:.1f}s")
            time.sleep(0.02)
        exc = future.exception()
        if exc is not None:
            raise RuntimeError(f"{label} transport error: {exc}") from exc
        return future.result()

    def _get_place_parameters(self, goal_handle) -> Dict[str, ParameterValue]:
        names = [
            "destination_query",
            *PROFILE_PARAMETER_KEYS,
            "verify_source_query",
        ]
        req = GetParameters.Request()
        req.names = names
        future = self._place_get_client.call_async(req)
        response = self._wait_future(
            future,
            float(self.p["parameter_service_timeout_s"]),
            goal_handle,
            "PlaceExecutor get_parameters",
        )
        if len(response.values) != len(names):
            raise RuntimeError("PlaceExecutor get_parameters returned unexpected value count")
        return {name: copy.deepcopy(value) for name, value in zip(names, response.values)}

    @staticmethod
    def _value_message(value: Any) -> ParameterValue:
        if isinstance(value, bool):
            return ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=bool(value))
        if isinstance(value, str):
            return ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
        if isinstance(value, int) and not isinstance(value, bool):
            return ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(value))
        if isinstance(value, (float, int)):
            return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        raise TypeError(f"Unsupported ROS parameter value type: {type(value).__name__}")

    def _apply_destination_profile(
        self,
        profile: DestinationProfile,
        source_query: str,
        goal_handle,
    ) -> Dict[str, Any]:
        if self._place_baseline is None:
            self._place_baseline = self._get_place_parameters(goal_handle)

        params: Dict[str, ParameterValue] = {}
        # Semantic query and physical geometry are intentionally decoupled.
        # Unknown destination text is preserved and sent directly to Grounded-SAM2;
        # only the physical place geometry falls back to the calibrated baseline.
        params["destination_query"] = self._value_message(profile.query)
        params["verify_source_query"] = self._value_message(source_query)

        applied_readable: Dict[str, Any] = {
            "requested_query": profile.query,
            "geometry_policy": "override" if profile.matched_override else "generic-startup-baseline",
            "geometry_name": profile.name if profile.matched_override else "<generic-default>",
            "inherit_place_baseline": profile.inherit_place_baseline,
        }

        for key in PROFILE_PARAMETER_KEYS:
            explicit = profile.values.get(key)
            if explicit is None:
                if profile.inherit_place_baseline:
                    params[key] = copy.deepcopy(self._place_baseline[key])
                    applied_readable[key] = "<startup-place-baseline>"
                elif profile.require_explicit_geometry:
                    raise RuntimeError(
                        f"Destination geometry '{profile.name}' requires explicit '{key}'"
                    )
                else:
                    params[key] = copy.deepcopy(self._place_baseline[key])
                    applied_readable[key] = "<startup-place-baseline>"
            else:
                params[key] = self._value_message(explicit)
                applied_readable[key] = explicit

        # Safety validation before writing parameters.
        rim_value = params["destination_rim_height_m"]
        rim = float(rim_value.double_value)
        if rim <= 0.0 or rim > 0.30:
            raise RuntimeError(
                f"Destination place geometry has unsafe rim height {rim:.4f}m for query "
                f"'{profile.query}'. Save the validated generic geometry in place.yaml or "
                "provide an optional override in destination_profiles.yaml."
            )

        req = SetParameters.Request()
        req.parameters = [ParameterMsg(name=name, value=value) for name, value in params.items()]
        future = self._place_set_client.call_async(req)
        response = self._wait_future(
            future,
            float(self.p["parameter_service_timeout_s"]),
            goal_handle,
            "PlaceExecutor set_parameters",
        )
        if len(response.results) != len(req.parameters):
            raise RuntimeError("PlaceExecutor set_parameters returned unexpected result count")
        failures = [
            f"{param.name}: {res.reason}"
            for param, res in zip(req.parameters, response.results)
            if not res.successful
        ]
        if failures:
            raise RuntimeError("Failed to apply destination place geometry: " + "; ".join(failures))
        return applied_readable

    # ------------------------- cancellation / abort -------------------------
    def _check_cancel(self, goal_handle, terminate_process: Optional[subprocess.Popen] = None) -> None:
        if not goal_handle.is_cancel_requested:
            return
        if terminate_process is not None and terminate_process.poll() is None:
            terminate_process.terminate()
        self._request_abort_no_wait()
        raise TaskCancelled("Task canceled by client; software abort requested")

    def _request_abort_no_wait(self) -> None:
        # SIGINT can invalidate the ROS context before the executor finally block runs.
        # In that state even service_is_ready() and rosout logging can raise RCLError.
        if not rclpy.ok():
            return
        for key in ("home_abort_service", "pick_abort_service", "place_abort_service"):
            client = self._trigger_clients.get(key)
            if client is None:
                continue
            try:
                if client.service_is_ready():
                    client.call_async(Trigger.Request())
            except Exception as exc:
                if rclpy.ok():
                    try:
                        self.get_logger().error(
                            f"Abort request failed for {self.p.get(key)}: {exc}"
                        )
                    except Exception:
                        pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RearrangementTaskManager()
    executor = MultiThreadedExecutor(num_threads=10)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node._request_abort_no_wait()
            except Exception:
                pass
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
