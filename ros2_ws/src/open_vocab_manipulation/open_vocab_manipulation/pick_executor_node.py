#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .geometry import euclidean2, euclidean3, pose_error, pose_spec_to_msg, translated_pose
from .gripper_client import GripperResult, RobotiqGripperClient
from .io_utils import file_age_s, read_json
from .journal import ExecutionJournal
from .local_perception_client import LocalPerceptionClient
from .models import GraspPlan, LocalObservation, PickState
from .moveit_client import MoveItExecutionClient


class PickExecutorNode(Node):
    """Safety-gated real execution layer for the existing PLAN-ONLY grasp planner.

    Safety-gated real execution layer for the existing PLAN-ONLY grasp planner.

    Two operating modes are supported:
      - fast static demo (require_source_revalidation=false):
        prepare -> move_pregrasp -> [approach_dry_run] -> grasp_and_lift
      - dynamic/revalidation mode (require_source_revalidation=true):
        prepare -> move_pregrasp -> revalidate -> [approach_dry_run] -> grasp_and_lift

    The revalidation service is intentionally retained but is optional in fast demo mode.
    """

    def __init__(self):
        super().__init__("open_vocab_pick_executor")
        self.cb_group = ReentrantCallbackGroup()
        self._declare_parameters()
        self.p = self._load_params()
        self._op_lock = threading.RLock()
        self.state = PickState.IDLE
        self.plan: Optional[GraspPlan] = None
        self.plan_file_mtime_ns: Optional[int] = None
        self.baseline_observation: Optional[LocalObservation] = None
        self.last_revalidation: Optional[LocalObservation] = None
        self.last_revalidation_monotonic = 0.0
        self.last_gripper_result: Optional[GripperResult] = None

        self.moveit = MoveItExecutionClient(self, self.cb_group, self.p)
        self.gripper = RobotiqGripperClient(
            self,
            self.cb_group,
            str(self.p["gripper_action"]),
            float(self.p["gripper_timeout_s"]),
        )
        self.local = LocalPerceptionClient(
            self,
            self.cb_group,
            str(self.p["local_perception_service"]),
            str(self.p["local_result_json_path"]),
        )
        self.journal = ExecutionJournal(
            str(self.p["latest_execution_json_path"]),
            str(self.p["execution_history_jsonl_path"]),
        )

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(
            String, str(self.p["status_topic"]), status_qos
        )

        self._create_services()
        self._publish_status("startup", {})
        mode = "DYNAMIC_REVALIDATION" if bool(self.p["require_source_revalidation"]) else "FAST_STATIC_DEMO"
        self.get_logger().info(
            f"PickExecutor ready. mode={mode}. "
            "Flow: prepare -> move_pregrasp -> [revalidate optional] -> "
            "[approach_dry_run optional] -> grasp_and_lift."
        )
        if not bool(self.p["require_source_revalidation"]):
            self.get_logger().warn(
                "Source revalidation is DISABLED. The object must remain static after the baseline local perception."
            )
        if not bool(self.p["enable_arm_execution"]):
            self.get_logger().warn(
                "ARM EXECUTION IS LOCKED (enable_arm_execution=false). "
                "This is the safe default."
            )
        if not bool(self.p["allow_approach_dry_run"]):
            self.get_logger().warn("Cartesian dry-run is locked (allow_approach_dry_run=false).")
        if not bool(self.p["enable_grasp_and_lift"]):
            self.get_logger().warn("Grasp-and-lift is locked (enable_grasp_and_lift=false).")

    def _declare_parameters(self) -> None:
        d = self.declare_parameter
        d("grasp_plan_json_path", "/opt/open-vocabulary-robot-rearrangement/perception/grasp_plans/latest_grasp_plan.json")
        d("local_result_json_path", "/opt/open-vocabulary-robot-rearrangement/perception/local_outputs/latest_local_result.json")
        d("max_plan_age_s", 120.0)
        d("max_local_result_age_s", 120.0)
        d("max_revalidation_age_s", 3.0)
        d("require_source_revalidation", False)

        d("base_frame", "base_link")
        d("planning_group", "manipulator")
        d("goal_link", "end_effector_link")
        d("joint_states_topic", "/joint_states")
        d("arm_joint_names", ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"])
        d("compute_ik_service", "/compute_ik")
        d("motion_plan_service", "/plan_kinematic_path")
        d("cartesian_path_service", "/compute_cartesian_path")
        d("get_planning_scene_service", "/get_planning_scene")
        d("state_validity_service", "/check_state_validity")
        d("execute_trajectory_action", "/execute_trajectory")
        d("trajectory_execution_event_topic", "/trajectory_execution_event")
        d("require_state_validity_service", True)
        d("table_object_id", "grasp_table")
        d("planner_id", "RRTConnectkConfigDefault")
        d("planning_attempts", 3)
        d("allowed_planning_time_s", 3.0)
        d("ik_timeout_s", 0.35)
        d("velocity_scaling", 0.06)
        d("acceleration_scaling", 0.05)
        d("cartesian_velocity_scaling", 0.03)
        d("cartesian_acceleration_scaling", 0.03)
        d("cartesian_max_step_m", 0.003)
        d("cartesian_jump_threshold", 0.0)
        d("min_cartesian_fraction", 0.995)
        d("max_single_joint_delta_rad", 2.80)
        d("max_joint_l1_delta_rad", 8.50)
        d("max_joint_state_age_s", 0.25)
        d("execution_timeout_s", 45.0)

        d("enable_arm_execution", False)
        d("allow_approach_dry_run", False)
        d("enable_gripper_execution", False)
        d("enable_grasp_and_lift", False)
        d("dry_run_open_gripper", True)

        d("max_gripper_width_m", 0.085)
        d("min_gripper_width_reserve_m", 0.004)
        d("gripper_action", "/robotiq_gripper_controller/gripper_cmd")
        d("gripper_open_position", 0.0)
        d("gripper_close_position", 0.80)
        d("gripper_max_effort", 40.0)
        d("gripper_timeout_s", 8.0)
        d("empty_close_position_threshold", 0.76)
        d("require_gripper_object_evidence", False)

        d("source_revalidate_max_xy_m", 0.025)
        d("source_revalidate_max_xyz_m", 0.035)
        d("source_revalidate_min_valid_depth_ratio", 0.80)
        d("source_revalidate_min_cloud_points", 150)
        d("local_perception_service", "/open_vocab/local_perception/trigger")
        d("local_perception_timeout_s", 15.0)

        d("pregrasp_position_tolerance_m", 0.015)
        d("grasp_position_tolerance_m", 0.012)
        d("orientation_tolerance_deg", 10.0)
        d("lift_distance_m", 0.10)
        d("lift_position_tolerance_m", 0.020)

        d("status_topic", "/open_vocab/pick/status")
        d("latest_execution_json_path", "/opt/open-vocabulary-robot-rearrangement/perception/manipulation_logs/latest_pick_execution.json")
        d("execution_history_jsonl_path", "/opt/open-vocabulary-robot-rearrangement/perception/manipulation_logs/pick_execution_history.jsonl")

    def _load_params(self) -> Dict[str, object]:
        names = [
            "grasp_plan_json_path", "local_result_json_path", "max_plan_age_s",
            "max_local_result_age_s", "max_revalidation_age_s", "require_source_revalidation", "base_frame",
            "planning_group", "goal_link", "joint_states_topic", "arm_joint_names",
            "compute_ik_service", "motion_plan_service", "cartesian_path_service",
            "get_planning_scene_service", "state_validity_service",
            "execute_trajectory_action", "trajectory_execution_event_topic",
            "require_state_validity_service", "table_object_id", "planner_id",
            "planning_attempts", "allowed_planning_time_s", "ik_timeout_s",
            "velocity_scaling", "acceleration_scaling", "cartesian_velocity_scaling",
            "cartesian_acceleration_scaling", "cartesian_max_step_m",
            "cartesian_jump_threshold", "min_cartesian_fraction",
            "max_single_joint_delta_rad", "max_joint_l1_delta_rad",
            "max_joint_state_age_s", "execution_timeout_s", "enable_arm_execution",
            "allow_approach_dry_run", "enable_gripper_execution", "enable_grasp_and_lift", "dry_run_open_gripper",
            "max_gripper_width_m", "min_gripper_width_reserve_m", "gripper_action",
            "gripper_open_position", "gripper_close_position", "gripper_max_effort",
            "gripper_timeout_s", "empty_close_position_threshold",
            "require_gripper_object_evidence", "source_revalidate_max_xy_m",
            "source_revalidate_max_xyz_m", "source_revalidate_min_valid_depth_ratio",
            "source_revalidate_min_cloud_points", "local_perception_service",
            "local_perception_timeout_s", "pregrasp_position_tolerance_m",
            "grasp_position_tolerance_m", "orientation_tolerance_deg", "lift_distance_m",
            "lift_position_tolerance_m", "status_topic", "latest_execution_json_path",
            "execution_history_jsonl_path",
        ]
        return {name: self.get_parameter(name).value for name in names}

    def _create_services(self) -> None:
        services = {
            "/open_vocab/pick/prepare": self._srv_prepare,
            "/open_vocab/pick/move_pregrasp": self._srv_move_pregrasp,
            "/open_vocab/pick/revalidate": self._srv_revalidate,
            "/open_vocab/pick/approach_dry_run": self._srv_approach_dry_run,
            "/open_vocab/pick/retreat_pregrasp": self._srv_retreat_pregrasp,
            "/open_vocab/pick/grasp_and_lift": self._srv_grasp_and_lift,
            "/open_vocab/pick/abort": self._srv_abort,
            "/open_vocab/pick/reset": self._srv_reset,
            "/open_vocab/pick/status": self._srv_status,
        }
        for name, cb in services.items():
            self.create_service(Trigger, name, cb, callback_group=self.cb_group)

    def _set_state(self, state: PickState, event: str, details: Dict[str, Any]) -> None:
        self.state = state
        self._publish_status(event, details)

    def _publish_status(self, event: str, details: Dict[str, Any]) -> None:
        payload = {
            "state": self.state.value,
            "event": event,
            "label": self.plan.label if self.plan else None,
            "object_id": self.plan.object_id if self.plan else None,
            "timestamp_unix_s": time.time(),
            "details": details,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        try:
            self.journal.write(event, self.state.value, payload)
        except Exception as exc:
            self.get_logger().error(f"Execution journal write failed: {exc}")

    def _service_guard(self, fn, response):
        try:
            with self._op_lock:
                message = fn()
            response.success = True
            response.message = message
        except Exception as exc:
            if self.state not in (PickState.INVALIDATED, PickState.ABORTED):
                self._set_state(
                    PickState.ERROR,
                    "error",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
            self.get_logger().error(
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
        return response

    def _load_plan_and_baseline(self) -> Tuple[GraspPlan, LocalObservation, int]:
        data, stat_result = read_json(str(self.p["grasp_plan_json_path"]))
        age = (
            time.time() - float(data["created_at_unix_s"])
            if data.get("created_at_unix_s") is not None
            else file_age_s(stat_result)
        )
        if age > float(self.p["max_plan_age_s"]):
            raise RuntimeError(
                f"Grasp plan stale: age={age:.1f}s > {float(self.p['max_plan_age_s']):.1f}s. "
                "Run /open_vocab/grasp/plan_latest again."
            )
        plan = GraspPlan.from_dict(data)
        if plan.base_frame != str(self.p["base_frame"]):
            raise RuntimeError(f"Plan base_frame={plan.base_frame}, expected {self.p['base_frame']}")
        if plan.planning_group != str(self.p["planning_group"]):
            raise RuntimeError(
                f"Plan planning_group={plan.planning_group}, expected {self.p['planning_group']}"
            )
        if plan.goal_link != str(self.p["goal_link"]):
            raise RuntimeError(f"Plan goal_link={plan.goal_link}, expected {self.p['goal_link']}")
        if not plan.table_collision_enabled:
            raise RuntimeError("Grasp plan was generated without table collision enabled")
        if plan.table_object_id != str(self.p["table_object_id"]):
            raise RuntimeError(
                f"Plan table object id={plan.table_object_id}, expected {self.p['table_object_id']}"
            )

        max_safe_width = float(self.p["max_gripper_width_m"]) - float(
            self.p["min_gripper_width_reserve_m"]
        )
        if plan.selected.width_m > max_safe_width:
            raise RuntimeError(
                f"Selected grasp width={plan.selected.width_m*1000:.1f}mm leaves insufficient "
                f"2F-85 reserve; safe limit={max_safe_width*1000:.1f}mm. Choose another object/candidate."
            )
        if plan.selected.cartesian_fraction < float(self.p["min_cartesian_fraction"]):
            raise RuntimeError(
                f"Plan Cartesian fraction={plan.selected.cartesian_fraction:.4f} below execution threshold"
            )

        local_data, local_stat = read_json(str(self.p["local_result_json_path"]))
        if file_age_s(local_stat) > float(self.p["max_local_result_age_s"]):
            raise RuntimeError("Baseline local perception result is stale")
        obs = LocalObservation.from_dict(local_data)
        self._assert_same_object(plan, obs)
        if plan.source_centroid_base_xyz_m is not None:
            baseline_delta = euclidean3(plan.source_centroid_base_xyz_m, obs.centroid_base_xyz_m)
            if baseline_delta > 0.020:
                raise RuntimeError(
                    f"Plan/local baseline mismatch: centroid delta={baseline_delta*1000:.1f}mm"
                )
        return plan, obs, stat_result.st_mtime_ns

    @staticmethod
    def _assert_same_object(plan: GraspPlan, obs: LocalObservation) -> None:
        if plan.object_id and obs.object_id and plan.object_id != obs.object_id:
            raise RuntimeError(
                f"Object ID mismatch: plan={plan.object_id}, local={obs.object_id}"
            )
        if plan.label.casefold() != obs.label.casefold():
            raise RuntimeError(f"Label mismatch: plan={plan.label}, local={obs.label}")

    def _assert_plan_unchanged(self) -> None:
        if self.plan_file_mtime_ns is None:
            raise RuntimeError("No prepared plan fingerprint")
        stat = Path(str(self.p["grasp_plan_json_path"])).stat()
        if stat.st_mtime_ns != self.plan_file_mtime_ns:
            raise RuntimeError(
                "latest_grasp_plan.json changed after prepare; reset and prepare the new plan"
            )

    def _assert_table_and_state(self) -> None:
        self.moveit.assert_current_state_valid()
        table_id = self.plan.table_object_id if self.plan else str(self.p["table_object_id"])
        if not self.moveit.table_exists(table_id):
            raise RuntimeError(
                f"Required PlanningScene table '{table_id}' is missing. "
                "Run /open_vocab/grasp/validate_table or plan_latest first."
            )

    def _assert_pose_reached(self, target, pos_tol_m: float) -> Tuple[float, float]:
        actual = self.moveit.current_goal_link_pose(timeout_s=1.0)
        pos_err, ori_err = pose_error(actual, target)
        ori_tol = math.radians(float(self.p["orientation_tolerance_deg"]))
        if pos_err > pos_tol_m or ori_err > ori_tol:
            raise RuntimeError(
                f"Final EEF pose outside tolerance: position_error={pos_err*1000:.1f}mm "
                f"(limit {pos_tol_m*1000:.1f}mm), orientation_error={math.degrees(ori_err):.1f}deg "
                f"(limit {self.p['orientation_tolerance_deg']}deg)"
            )
        return pos_err, ori_err

    def _recent_revalidation_required(self) -> None:
        if self.state != PickState.REVALIDATED:
            raise RuntimeError(f"Expected state REVALIDATED, current state={self.state.value}")
        age = time.monotonic() - self.last_revalidation_monotonic
        if age > float(self.p["max_revalidation_age_s"]):
            self._set_state(PickState.AT_PREGRASP, "revalidation_expired", {"age_s": age})
            raise RuntimeError(
                f"Revalidation expired: age={age:.2f}s > {float(self.p['max_revalidation_age_s']):.2f}s; "
                "call /open_vocab/pick/revalidate again"
            )

    def _assert_approach_gate(self, *, allow_grasp_dry_pose: bool = False) -> None:
        """Gate Cartesian approach/grasp according to the selected demo mode.

        Dynamic mode preserves the original mandatory revalidation behavior.
        Fast static demo mode intentionally skips the second wrist-camera observation,
        but still requires the executor to be at the planned pregrasp (or, when
        explicitly allowed, at a previously verified dry grasp pose).
        """
        if bool(self.p["require_source_revalidation"]):
            self._recent_revalidation_required()
            return

        allowed = {PickState.AT_PREGRASP, PickState.REVALIDATED}
        if allow_grasp_dry_pose:
            allowed.add(PickState.AT_GRASP_DRY)
        if self.state not in allowed:
            names = ", ".join(sorted(s.value for s in allowed))
            raise RuntimeError(
                f"Fast static approach requires one of [{names}], current={self.state.value}"
            )

    def _gripper_object_evidence(self, result) -> bool:
        """Return conservative evidence that the gripper stopped on an object.

        Kortex/GripperCommand uses an approximately 0.0=open -> 0.8=closed
        controller coordinate.  An empty close should finish near the configured
        fully-closed position; an object typically causes an earlier stop and/or
        a stalled result.  This helper is telemetry by default because
        require_gripper_object_evidence=false in FAST_STATIC_DEMO.
        """
        threshold = float(self.p["empty_close_position_threshold"])
        return bool(result.stalled) or float(result.position) < threshold

    # ----------------------------- Services -----------------------------
    def _srv_prepare(self, _, response):
        def op():
            if self.state not in (PickState.IDLE, PickState.ERROR, PickState.ABORTED, PickState.INVALIDATED):
                raise RuntimeError(f"prepare not allowed from state {self.state.value}; reset first")
            self.get_logger().info("prepare[1/3]: checking MoveIt core services (no motion)...")
            # PREPARE is a non-motion safety gate. Do not require the real
            # /execute_trajectory action server here; that is checked immediately
            # before actual execution by execute_trajectory().
            self.moveit.wait_until_ready(3.0, require_execute=False)
            self.get_logger().info("prepare[2/3]: loading fresh grasp plan and local baseline...")
            plan, baseline, mtime_ns = self._load_plan_and_baseline()
            self.plan = plan
            self.baseline_observation = baseline
            self.plan_file_mtime_ns = mtime_ns
            self.last_revalidation = None
            self.last_revalidation_monotonic = 0.0
            self.get_logger().info("prepare[3/3]: validating current robot state and planning scene...")
            self._assert_table_and_state()
            details = {
                "label": plan.label,
                "object_id": plan.object_id,
                "selected_rank": plan.selected.rank,
                "grasp_width_m": plan.selected.width_m,
                "cartesian_fraction": plan.selected.cartesian_fraction,
                "baseline_centroid_base_xyz_m": baseline.centroid_base_xyz_m,
                "arm_execution_enabled": bool(self.p["enable_arm_execution"]),
            }
            self._set_state(PickState.PREPARED, "prepared", details)
            return (
                f"PREPARED: {plan.label}, rank={plan.selected.rank}, "
                f"width={plan.selected.width_m*1000:.1f}mm. No motion executed."
            )
        return self._service_guard(op, response)

    def _srv_move_pregrasp(self, _, response):
        def op():
            if self.state != PickState.PREPARED:
                raise RuntimeError(f"move_pregrasp requires PREPARED, current={self.state.value}")
            if not bool(self.p["enable_arm_execution"]):
                raise RuntimeError("Arm execution locked: set enable_arm_execution=true")
            self._assert_plan_unchanged()
            self._assert_table_and_state()
            target = pose_spec_to_msg(self.plan.selected.pregrasp_pose, self.get_clock().now().to_msg())
            result, max_delta, l1 = self.moveit.plan_to_pose_from_current(target)
            self._set_state(
                PickState.MOVING_PREGRASP,
                "pregrasp_execution_started",
                {"max_joint_delta_rad": max_delta, "joint_l1_delta_rad": l1},
            )
            self.moveit.execute_trajectory(result.trajectory)
            pos_err, ori_err = self._assert_pose_reached(
                target, float(self.p["pregrasp_position_tolerance_m"])
            )
            self._set_state(
                PickState.AT_PREGRASP,
                "pregrasp_reached",
                {"position_error_m": pos_err, "orientation_error_rad": ori_err},
            )
            return (
                f"AT_PREGRASP: final error={pos_err*1000:.1f}mm / "
                f"{math.degrees(ori_err):.1f}deg. Inspect robot, target and table clearance now."
            )
        return self._service_guard(op, response)

    def _srv_revalidate(self, _, response):
        def op():
            if self.state not in (PickState.AT_PREGRASP, PickState.REVALIDATED):
                raise RuntimeError(f"revalidate requires AT_PREGRASP, current={self.state.value}")
            self._assert_plan_unchanged()
            self._assert_table_and_state()
            self._set_state(PickState.REVALIDATING, "revalidation_started", {})
            obs = self.local.trigger_and_read(float(self.p["local_perception_timeout_s"]))
            self._assert_same_object(self.plan, obs)
            if obs.valid_depth_ratio is not None and obs.valid_depth_ratio < float(
                self.p["source_revalidate_min_valid_depth_ratio"]
            ):
                raise RuntimeError(
                    f"Revalidation depth ratio={obs.valid_depth_ratio:.3f} below threshold"
                )
            if obs.cloud_points is not None and obs.cloud_points < int(
                self.p["source_revalidate_min_cloud_points"]
            ):
                raise RuntimeError(
                    f"Revalidation point cloud too small: {obs.cloud_points} points"
                )
            baseline = self.baseline_observation.centroid_base_xyz_m
            xy = euclidean2(baseline, obs.centroid_base_xyz_m)
            xyz = euclidean3(baseline, obs.centroid_base_xyz_m)
            if xy > float(self.p["source_revalidate_max_xy_m"]) or xyz > float(
                self.p["source_revalidate_max_xyz_m"]
            ):
                self.last_revalidation = obs
                self._set_state(
                    PickState.INVALIDATED,
                    "source_moved",
                    {
                        "baseline_centroid": baseline,
                        "current_centroid": obs.centroid_base_xyz_m,
                        "xy_displacement_m": xy,
                        "xyz_displacement_m": xyz,
                    },
                )
                raise RuntimeError(
                    f"SOURCE MOVED: xy={xy*1000:.1f}mm, xyz={xyz*1000:.1f}mm. "
                    "Old grasp is invalid. Re-run local perception + grasp planning, then reset/prepare."
                )
            self.last_revalidation = obs
            self.last_revalidation_monotonic = time.monotonic()
            self._set_state(
                PickState.REVALIDATED,
                "source_revalidated",
                {"xy_displacement_m": xy, "xyz_displacement_m": xyz},
            )
            return f"REVALIDATED: source movement xy={xy*1000:.1f}mm, xyz={xyz*1000:.1f}mm"
        return self._service_guard(op, response)

    def _srv_approach_dry_run(self, _, response):
        def op():
            self._assert_approach_gate(allow_grasp_dry_pose=False)
            if not bool(self.p["enable_arm_execution"]):
                raise RuntimeError("Arm execution locked: enable_arm_execution=false")
            if not bool(self.p["allow_approach_dry_run"]):
                raise RuntimeError("Dry-run locked: set allow_approach_dry_run=true only after pregrasp inspection")
            self._assert_plan_unchanged()
            self._assert_table_and_state()
            if bool(self.p["dry_run_open_gripper"]):
                if not bool(self.p["enable_gripper_execution"]):
                    raise RuntimeError(
                        "dry_run_open_gripper=true but enable_gripper_execution=false. "
                        "Enable gripper execution or set dry_run_open_gripper=false."
                    )
                self.gripper.wait_until_ready(3.0)
                self.gripper.command(
                    float(self.p["gripper_open_position"]),
                    float(self.p["gripper_max_effort"]),
                )
            target = pose_spec_to_msg(self.plan.selected.grasp_pose, self.get_clock().now().to_msg())
            trajectory, fraction = self.moveit.compute_cartesian_from_current(target)
            self._set_state(
                PickState.APPROACHING_DRY,
                "dry_approach_started",
                {"cartesian_fraction": fraction},
            )
            self.moveit.execute_trajectory(trajectory)
            pos_err, ori_err = self._assert_pose_reached(
                target, float(self.p["grasp_position_tolerance_m"])
            )
            self._set_state(
                PickState.AT_GRASP_DRY,
                "dry_grasp_pose_reached",
                {"position_error_m": pos_err, "orientation_error_rad": ori_err},
            )
            return (
                f"AT_GRASP_DRY: NO GRIPPER COMMAND SENT. error={pos_err*1000:.1f}mm / "
                f"{math.degrees(ori_err):.1f}deg. Inspect finger/table/object clearance."
            )
        return self._service_guard(op, response)

    def _srv_retreat_pregrasp(self, _, response):
        def op():
            if self.state not in (PickState.AT_GRASP_DRY, PickState.REVALIDATED):
                raise RuntimeError(
                    f"retreat_pregrasp requires AT_GRASP_DRY/REVALIDATED, current={self.state.value}"
                )
            if not bool(self.p["enable_arm_execution"]):
                raise RuntimeError("Arm execution locked")
            target = pose_spec_to_msg(self.plan.selected.pregrasp_pose, self.get_clock().now().to_msg())
            trajectory, fraction = self.moveit.compute_cartesian_from_current(target)
            self._set_state(PickState.RETREATING, "retreat_started", {"cartesian_fraction": fraction})
            self.moveit.execute_trajectory(trajectory)
            pos_err, ori_err = self._assert_pose_reached(
                target, float(self.p["pregrasp_position_tolerance_m"])
            )
            self.last_revalidation_monotonic = 0.0
            self._set_state(
                PickState.AT_PREGRASP,
                "retreated_to_pregrasp",
                {"position_error_m": pos_err, "orientation_error_rad": ori_err},
            )
            return "Retreated safely to pregrasp. Revalidate again before any new approach."
        return self._service_guard(op, response)

    def _srv_grasp_and_lift(self, _, response):
        def op():
            self._assert_approach_gate(allow_grasp_dry_pose=True)
            if not bool(self.p["enable_arm_execution"]):
                raise RuntimeError("Arm execution locked")
            if not bool(self.p["enable_gripper_execution"]):
                raise RuntimeError("Gripper execution locked: enable_gripper_execution=false")
            if not bool(self.p["enable_grasp_and_lift"]):
                raise RuntimeError(
                    "grasp_and_lift locked. Enable only after pregrasp/dry-run geometry has been verified."
                )
            self._assert_plan_unchanged()
            self._assert_table_and_state()
            self.gripper.wait_until_ready(3.0)

            # Two supported entry points:
            #   AT_PREGRASP/REVALIDATED -> open, Cartesian approach, close, lift
            #   AT_GRASP_DRY            -> dry-run already verified/reached the exact grasp pose;
            #                              close in place, then lift (no duplicate approach).
            already_at_grasp = self.state == PickState.AT_GRASP_DRY
            target = pose_spec_to_msg(self.plan.selected.grasp_pose, self.get_clock().now().to_msg())

            if already_at_grasp:
                pos_err, ori_err = self._assert_pose_reached(
                    target, float(self.p["grasp_position_tolerance_m"])
                )
                self._set_state(
                    PickState.GRASPING,
                    "real_grasp_from_dry_pose",
                    {
                        "position_error_m": pos_err,
                        "orientation_error_rad": ori_err,
                        "approach_skipped": True,
                    },
                )
            else:
                open_result = self.gripper.command(
                    float(self.p["gripper_open_position"]),
                    float(self.p["gripper_max_effort"]),
                )
                approach_traj, fraction = self.moveit.compute_cartesian_from_current(target)
                self._set_state(
                    PickState.GRASPING,
                    "real_approach_started",
                    {"cartesian_fraction": fraction, "approach_skipped": False},
                )
                self.moveit.execute_trajectory(approach_traj)
                self._assert_pose_reached(target, float(self.p["grasp_position_tolerance_m"]))

            close_result = self.gripper.command(
                float(self.p["gripper_close_position"]),
                float(self.p["gripper_max_effort"]),
            )
            self.last_gripper_result = close_result
            evidence = self._gripper_object_evidence(close_result)
            if bool(self.p["require_gripper_object_evidence"]) and not evidence:
                raise RuntimeError(
                    "Gripper closed without object-contact evidence; lift suppressed by safety policy"
                )

            current_pose = self.moveit.current_goal_link_pose(timeout_s=1.0)
            lift_target = translated_pose(current_pose, float(self.p["lift_distance_m"]))
            lift_traj, lift_fraction = self.moveit.compute_cartesian_from_current(lift_target)
            self._set_state(
                PickState.LIFTING,
                "lift_started",
                {
                    "cartesian_fraction": lift_fraction,
                    "gripper": close_result.__dict__,
                    "object_evidence": evidence,
                    "started_from_dry_pose": already_at_grasp,
                },
            )
            self.moveit.execute_trajectory(lift_traj)
            lift_err, _ = self._assert_pose_reached(
                lift_target, float(self.p["lift_position_tolerance_m"])
            )
            self._set_state(
                PickState.PICKED,
                "pick_lift_completed",
                {
                    "lift_position_error_m": lift_err,
                    "gripper": close_result.__dict__,
                    "object_evidence": evidence,
                    "started_from_dry_pose": already_at_grasp,
                },
            )
            return (
                f"PICKED: lifted {float(self.p['lift_distance_m'])*1000:.0f}mm. "
                f"gripper_position={close_result.position:.3f}, stalled={close_result.stalled}, "
                f"reached_goal={close_result.reached_goal}, object_evidence={evidence}, "
                f"started_from_dry_pose={already_at_grasp}."
            )
        return self._service_guard(op, response)

    def _srv_abort(self, _, response):
        try:
            self.moveit.stop()
            self.gripper.cancel()
            self._set_state(PickState.ABORTED, "abort_requested", {})
            response.success = True
            response.message = (
                "Software abort requested: MoveIt trajectory stop + active action cancellation sent. "
                "Use the physical E-stop for any unsafe robot motion."
            )
        except Exception as exc:
            response.success = False
            response.message = f"Abort encountered error: {exc}"
        return response

    def _srv_reset(self, _, response):
        def op():
            if self.state in (
                PickState.MOVING_PREGRASP,
                PickState.APPROACHING_DRY,
                PickState.RETREATING,
                PickState.GRASPING,
                PickState.LIFTING,
            ):
                raise RuntimeError("Cannot reset while a motion state is active; abort first")
            self.plan = None
            self.plan_file_mtime_ns = None
            self.baseline_observation = None
            self.last_revalidation = None
            self.last_revalidation_monotonic = 0.0
            self.last_gripper_result = None
            self._set_state(PickState.IDLE, "reset", {})
            return "Reset to IDLE."
        return self._service_guard(op, response)

    def _srv_status(self, _, response):
        details = {
            "state": self.state.value,
            "label": self.plan.label if self.plan else None,
            "object_id": self.plan.object_id if self.plan else None,
            "last_revalidation_age_s": (
                time.monotonic() - self.last_revalidation_monotonic
                if self.last_revalidation_monotonic > 0.0
                else None
            ),
            "arm_execution_enabled": bool(self.p["enable_arm_execution"]),
            "dry_run_enabled": bool(self.p["allow_approach_dry_run"]),
            "require_source_revalidation": bool(self.p["require_source_revalidation"]),
            "dry_run_open_gripper": bool(self.p["dry_run_open_gripper"]),
            "gripper_execution_enabled": bool(self.p["enable_gripper_execution"]),
            "grasp_and_lift_enabled": bool(self.p["enable_grasp_and_lift"]),
        }
        response.success = True
        response.message = json.dumps(details, ensure_ascii=False)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PickExecutorNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # On ROS 2 Humble SIGINT may already have shut down the context.
        pass
    finally:
        try:
            node.moveit.stop()
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
        # Avoid RCLError: "rcl_shutdown already called on the given context".
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
