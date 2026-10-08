#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
from visualization_msgs.msg import MarkerArray

from .geometry import (
    pose_error,
    pose_spec_to_msg,
    quaternion_angle_rad,
    quaternion_multiply_xyzw,
    quaternion_from_yaw,
    rotate_vector_about_z,
)
from .global_destination import DestinationObservation, GlobalDestinationLocator
from .gripper_client import RobotiqGripperClient
from .io_utils import atomic_write_json, read_json
from .journal import ExecutionJournal
from .models import GraspPlan, PoseSpec
from .moveit_client import MoveItExecutionClient
from .place_markers import make_place_markers
from .place_models import PlacePlan, PlaceState


class PlaceExecutorNode(Node):
    """Safety-gated destination localization and place execution.

    Fast demo policy:
      1) locate_destination: global HIK + Grounded-SAM2 + calibrated homography
      2) prepare: construct a receptacle-centered release pose, preview OMPL and
         Cartesian feasibility, publish RViz markers, but do not move
      3) move_preplace: real OMPL motion above the receptacle
      4) release_and_retreat: Cartesian descent, open 2F-85, Cartesian retreat

    The destination Z does not rely on D435i depth.  For transparent glass bowls,
    the user supplies the measured rim height once; global vision supplies XY.
    """

    def __init__(self):
        super().__init__("open_vocab_place_executor")
        self.cb_group = ReentrantCallbackGroup()
        self._declare_parameters()
        self.p = self._load_params()
        self._op_lock = threading.RLock()
        self.state = PlaceState.IDLE
        self.destination: Optional[DestinationObservation] = None
        self.plan: Optional[PlacePlan] = None
        self.destination_file_mtime_ns: Optional[int] = None

        self.moveit = MoveItExecutionClient(self, self.cb_group, self.p)
        self.gripper = RobotiqGripperClient(
            self,
            self.cb_group,
            str(self.p["gripper_action"]),
            float(self.p["gripper_timeout_s"]),
        )
        self.locator = GlobalDestinationLocator(
            base_url=str(self.p["ai_service_base_url"]),
            homography_npy_path=str(self.p["homography_npy_path"]),
            expected_image_space_id=str(self.p["expected_image_space_id"]),
            expected_width=int(self.p["expected_width"]),
            expected_height=int(self.p["expected_height"]),
            connect_timeout_s=float(self.p["ai_connect_timeout_s"]),
            read_timeout_s=float(self.p["ai_read_timeout_s"]),
            min_confidence=float(self.p["destination_min_confidence"]),
            min_mask_area_px=int(self.p["destination_min_mask_area_px"]),
            reject_ambiguous=bool(self.p["destination_reject_ambiguous"]),
            ambiguity_margin=float(self.p["destination_ambiguity_margin"]),
            workspace={
                "x_min": float(self.p["workspace_x_min_m"]),
                "x_max": float(self.p["workspace_x_max_m"]),
                "y_min": float(self.p["workspace_y_min_m"]),
                "y_max": float(self.p["workspace_y_max_m"]),
            },
        )
        self.journal = ExecutionJournal(
            str(self.p["latest_place_execution_json_path"]),
            str(self.p["place_execution_history_jsonl_path"]),
        )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(String, str(self.p["status_topic"]), qos)
        self.marker_pub = self.create_publisher(MarkerArray, str(self.p["marker_topic"]), qos)

        self._create_services()
        self._publish_status("startup", {})
        self.get_logger().info(
            "PlaceExecutor ready. Flow: locate_destination -> prepare -> "
            "move_preplace -> release_and_retreat -> [verify optional]."
        )
        self.get_logger().warn(
            "FAST PLACE uses global RGB+Homography for destination XY and a measured "
            "container rim height for Z. Keep the receptacle static during execution."
        )
        if not bool(self.p["enable_place_arm_execution"]):
            self.get_logger().warn("PLACE ARM EXECUTION IS LOCKED (enable_place_arm_execution=false).")
        if not bool(self.p["enable_release_execution"]):
            self.get_logger().warn("RELEASE IS LOCKED (enable_release_execution=false).")

    def _declare_parameters(self) -> None:
        d = self.declare_parameter
        # Existing artifacts / pick evidence
        d("grasp_plan_json_path", "/opt/open-vocabulary-robot-rearrangement/perception/grasp_plans/latest_grasp_plan.json")
        d("pick_history_jsonl_path", "/opt/open-vocabulary-robot-rearrangement/perception/manipulation_logs/pick_execution_history.jsonl")
        d("max_pick_age_s", 3600.0)
        d("require_recent_pick_evidence", True)

        # Destination global perception
        d("destination_query", "glass bowl")
        d("ai_service_base_url", "http://ai-server.local:8090")
        d("homography_npy_path", "/opt/open-vocabulary-robot-rearrangement/perception/calibration/H_image_to_base.npy")
        d("expected_image_space_id", "UNDISTORTED_1280x720")
        d("expected_width", 1280)
        d("expected_height", 720)
        d("ai_connect_timeout_s", 2.0)
        d("ai_read_timeout_s", 20.0)
        d("destination_min_confidence", 0.45)
        d("destination_min_mask_area_px", 300)
        d("destination_reject_ambiguous", True)
        d("destination_ambiguity_margin", 0.08)
        d("max_destination_age_s", 120.0)
        d("destination_xy_offset_x_m", 0.0)
        d("destination_xy_offset_y_m", 0.0)
        d("workspace_x_min_m", 0.12)
        d("workspace_x_max_m", 0.88)
        d("workspace_y_min_m", -0.40)
        d("workspace_y_max_m", 0.40)
        d("latest_destination_json_path", "/opt/open-vocabulary-robot-rearrangement/perception/place_outputs/latest_destination.json")
        d("latest_place_plan_json_path", "/opt/open-vocabulary-robot-rearrangement/perception/place_outputs/latest_place_plan.json")

        # Receptacle geometry. Measure rim height from the tabletop.
        d("destination_rim_height_m", -1.0)
        d("release_grasp_center_above_rim_m", 0.060)
        d("preplace_clearance_m", 0.100)
        # Robust place-pose search.  The held object may be transported with a
        # different yaw about base Z; this often recovers Kinova IK without
        # changing the grasp tilt or the grasp-center geometry.
        d("place_yaw_search_deg", [0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0])
        d("preplace_clearance_candidates_m", [0.100, 0.080, 0.120])
        d("place_candidate_yaw_penalty_rad_per_deg", 0.020)
        d("max_place_goal_z_m", 0.80)
        d("min_release_grasp_center_above_table_m", 0.050)

        # MoveIt / Kinova: same execution stack as pick.
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
        d("allowed_planning_time_s", 4.0)
        d("ik_timeout_s", 0.35)
        d("velocity_scaling", 0.06)
        d("acceleration_scaling", 0.05)
        d("cartesian_velocity_scaling", 0.025)
        d("cartesian_acceleration_scaling", 0.025)
        d("cartesian_max_step_m", 0.003)
        d("cartesian_jump_threshold", 0.0)
        d("min_cartesian_fraction", 0.995)
        d("max_single_joint_delta_rad", 2.80)
        d("max_joint_l1_delta_rad", 8.50)
        d("max_joint_state_age_s", 0.25)
        d("execution_timeout_s", 60.0)

        # Real execution gates.
        d("enable_place_arm_execution", False)
        d("enable_release_execution", False)
        d("gripper_action", "/robotiq_gripper_controller/gripper_cmd")
        d("gripper_open_position", 0.0)
        d("gripper_max_effort", 40.0)
        d("gripper_timeout_s", 8.0)
        d("release_dwell_s", 0.50)
        d("preplace_position_tolerance_m", 0.020)
        d("release_position_tolerance_m", 0.015)
        d("orientation_tolerance_deg", 12.0)

        # Optional coarse post-place verification.
        d("verify_acceptance_radius_m", 0.14)
        d("verify_source_query", "")

        # Audit / RViz.
        d("status_topic", "/open_vocab/place/status")
        d("marker_topic", "/open_vocab/place/markers")
        d("latest_place_execution_json_path", "/opt/open-vocabulary-robot-rearrangement/perception/manipulation_logs/latest_place_execution.json")
        d("place_execution_history_jsonl_path", "/opt/open-vocabulary-robot-rearrangement/perception/manipulation_logs/place_execution_history.jsonl")

    def _load_params(self) -> Dict[str, object]:
        names = list(self._parameters.keys())
        return {name: self.get_parameter(name).value for name in names}

    def _create_services(self) -> None:
        services = {
            "/open_vocab/place/locate_destination": self._srv_locate_destination,
            "/open_vocab/place/prepare": self._srv_prepare,
            "/open_vocab/place/move_preplace": self._srv_move_preplace,
            "/open_vocab/place/release_and_retreat": self._srv_release_and_retreat,
            "/open_vocab/place/verify": self._srv_verify,
            "/open_vocab/place/abort": self._srv_abort,
            "/open_vocab/place/reset": self._srv_reset,
            "/open_vocab/place/status": self._srv_status,
        }
        for name, cb in services.items():
            self.create_service(Trigger, name, cb, callback_group=self.cb_group)

    def _set_state(self, state: PlaceState, event: str, details: Dict[str, Any]) -> None:
        self.state = state
        self._publish_status(event, details)

    def _publish_status(self, event: str, details: Dict[str, Any]) -> None:
        payload = {
            "state": self.state.value,
            "event": event,
            "timestamp_unix_s": time.time(),
            "destination_query": self.destination.query if self.destination else None,
            "destination_label": self.destination.label if self.destination else None,
            "details": details,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        try:
            self.journal.write(event, self.state.value, payload)
        except Exception as exc:
            self.get_logger().error(f"Place journal write failed: {exc}")

    def _service_guard(self, fn, response):
        try:
            with self._op_lock:
                message = fn()
            response.success = True
            response.message = message
        except Exception as exc:
            if self.state != PlaceState.ABORTED:
                self._set_state(PlaceState.ERROR, "error", {"type": type(exc).__name__, "message": str(exc)})
            self.get_logger().error(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
        return response

    def _runtime_place_settings(self) -> Dict[str, Any]:
        yaw_values = [float(v) for v in self.get_parameter("place_yaw_search_deg").value]
        clearance_values = [float(v) for v in self.get_parameter("preplace_clearance_candidates_m").value]
        if not yaw_values:
            yaw_values = [0.0]
        if not clearance_values:
            clearance_values = [float(self.get_parameter("preplace_clearance_m").value)]
        return {
            "rim": float(self.get_parameter("destination_rim_height_m").value),
            "release_above_rim": float(self.get_parameter("release_grasp_center_above_rim_m").value),
            "preplace_clearance": float(self.get_parameter("preplace_clearance_m").value),
            "yaw_search_deg": yaw_values,
            "clearance_candidates_m": clearance_values,
            "yaw_penalty": float(self.get_parameter("place_candidate_yaw_penalty_rad_per_deg").value),
            "dx": float(self.get_parameter("destination_xy_offset_x_m").value),
            "dy": float(self.get_parameter("destination_xy_offset_y_m").value),
        }

    def _find_recent_pick(self, object_id: str, label: str) -> Dict[str, Any]:
        if not bool(self.p["require_recent_pick_evidence"]):
            return {"state": "UNENFORCED", "timestamp_unix_s": time.time()}
        path = Path(str(self.p["pick_history_jsonl_path"]))
        if not path.is_file():
            raise RuntimeError(f"Pick history not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines[-500:]):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if str(row.get("state")) != "PICKED" or str(row.get("event")) != "pick_lift_completed":
                continue
            age = time.time() - float(row.get("timestamp_unix_s", 0.0))
            if age > float(self.p["max_pick_age_s"]):
                raise RuntimeError(f"Latest PICKED evidence is stale: age={age:.1f}s")
            payload = row.get("details") or {}
            if object_id and payload.get("object_id") and str(payload["object_id"]) != object_id:
                continue
            if label and payload.get("label") and str(payload["label"]).casefold() != label.casefold():
                continue
            return row
        raise RuntimeError("No recent PICKED event matching the current grasp plan")

    def _load_grasp_context(self) -> Tuple[GraspPlan, Dict[str, Any], Dict[str, Any]]:
        raw, _ = read_json(str(self.p["grasp_plan_json_path"]))
        plan = GraspPlan.from_dict(raw)
        pick_row = self._find_recent_pick(plan.object_id, plan.label)
        if plan.base_frame != str(self.p["base_frame"]) or plan.goal_link != str(self.p["goal_link"]):
            raise RuntimeError("Grasp plan frame/link does not match PlaceExecutor configuration")
        if plan.physical_table_top_z_m is None:
            raise RuntimeError("Grasp plan does not contain physical_table_top_z_m")
        return plan, raw, pick_row

    def _load_destination(self) -> DestinationObservation:
        data, stat = read_json(str(self.p["latest_destination_json_path"]))
        obs = DestinationObservation.from_dict(data)
        age = time.time() - float(obs.created_at_unix_s)
        if age > float(self.p["max_destination_age_s"]):
            raise RuntimeError(
                f"Destination stale: age={age:.1f}s > {float(self.p['max_destination_age_s']):.1f}s; locate again"
            )
        current_query = str(self.get_parameter("destination_query").value).strip()
        if obs.query.casefold() != current_query.casefold():
            raise RuntimeError(
                f"Destination query changed: cached='{obs.query}', configured='{current_query}'. Locate again."
            )
        self.destination_file_mtime_ns = stat.st_mtime_ns
        return obs

    @staticmethod
    def _make_pose(frame_id: str, position, quaternion) -> PoseSpec:
        return PoseSpec(
            frame_id=frame_id,
            position=[float(v) for v in position],
            quaternion_xyzw=[float(v) for v in quaternion],
        )

    def _build_place_plan(
        self,
        grasp: GraspPlan,
        raw: Dict[str, Any],
        dest: DestinationObservation,
        yaw_offset_deg: float = 0.0,
        preplace_clearance_m: Optional[float] = None,
    ) -> PlacePlan:
        settings = self._runtime_place_settings()
        rim = settings["rim"]
        if rim <= 0.0 or rim > 0.30:
            raise RuntimeError(
                "destination_rim_height_m must be measured and set to a realistic positive value (metres), e.g. 0.085"
            )
        if settings["release_above_rim"] < 0.02:
            raise RuntimeError("release_grasp_center_above_rim_m is too small for the fast glass-bowl demo")
        clearance = (
            float(settings["preplace_clearance"])
            if preplace_clearance_m is None
            else float(preplace_clearance_m)
        )
        if clearance < 0.05:
            raise RuntimeError("preplace_clearance_m must be >= 0.05 m")

        selected = raw.get("selected") or {}
        approach = selected.get("approach_axis_base") or []
        if len(approach) != 3:
            raise RuntimeError("Grasp plan missing selected.approach_axis_base")
        d = float(raw.get("goal_link_to_grasp_center_m", 0.0))
        if d <= 0.05 or d >= 0.30:
            raise RuntimeError(f"Invalid goal_link_to_grasp_center_m={d}")
        # Original EEF->grasp-center offset is expressed in base frame at the
        # successful grasp orientation.  When we yaw the held object about base Z,
        # rotate BOTH the EEF orientation and this offset so the grasp center stays
        # exactly over the receptacle anchor.
        base_offset = [d * float(a) for a in approach]
        yaw_rad = math.radians(float(yaw_offset_deg))
        offset = rotate_vector_about_z(base_offset, yaw_rad)

        grasp_pose = grasp.selected.grasp_pose
        grasp_quat = list(grasp_pose.quaternion_xyzw)
        quat = quaternion_multiply_xyzw(quaternion_from_yaw(yaw_rad), grasp_quat)
        actual = self.moveit.current_goal_link_pose(timeout_s=1.0)
        q_now = [
            actual.pose.orientation.x,
            actual.pose.orientation.y,
            actual.pose.orientation.z,
            actual.pose.orientation.w,
        ]
        ori_err = quaternion_angle_rad(q_now, grasp_quat)
        if math.degrees(ori_err) > float(self.p["orientation_tolerance_deg"]):
            raise RuntimeError(
                f"Current held-object orientation differs from grasp orientation by {math.degrees(ori_err):.1f}deg; "
                "reset/recover before place"
            )

        x = float(dest.base_xy_m[0]) + settings["dx"]
        y = float(dest.base_xy_m[1]) + settings["dy"]
        table_z = float(grasp.physical_table_top_z_m)
        anchor_z = table_z + rim + settings["release_above_rim"]
        if anchor_z - table_z < float(self.p["min_release_grasp_center_above_table_m"]):
            raise RuntimeError("Release grasp-center height is too close to the tabletop")

        desired_anchor = [x, y, anchor_z]
        release_pos = [desired_anchor[i] - offset[i] for i in range(3)]
        preplace_pos = list(release_pos)
        preplace_pos[2] += clearance
        if max(release_pos[2], preplace_pos[2]) > float(self.p["max_place_goal_z_m"]):
            raise RuntimeError("Generated place pose exceeds configured max_place_goal_z_m")

        return PlacePlan(
            source_label=grasp.label,
            source_object_id=grasp.object_id,
            destination_query=dest.query,
            destination_label=dest.label,
            destination_xy_m=[x, y],
            desired_grasp_center_xyz_m=desired_anchor,
            grasp_center_offset_base_m=offset,
            preplace_pose=self._make_pose(str(self.p["base_frame"]), preplace_pos, quat),
            release_pose=self._make_pose(str(self.p["base_frame"]), release_pos, quat),
            table_top_z_m=table_z,
            destination_rim_height_m=rim,
            release_grasp_center_above_rim_m=settings["release_above_rim"],
            preplace_clearance_m=clearance,
        )

    def _search_feasible_place_plan(
        self, grasp: GraspPlan, raw: Dict[str, Any], dest: DestinationObservation
    ):
        """Search a small, deterministic family of place poses.

        The successful pick tilt is preserved.  We first try the original yaw,
        then increasingly larger base-Z yaw changes.  Within the first yaw group
        that has any safe solution, the clearance with the smallest joint L1 is
        selected.  This keeps object reorientation small and avoids spending
        unnecessary time planning all 24 combinations.
        """
        settings = self._runtime_place_settings()
        min_fraction = float(self.p["min_cartesian_fraction"])
        failures = []
        tested = 0

        for yaw_deg in settings["yaw_search_deg"]:
            feasible_this_yaw = []
            for clearance in settings["clearance_candidates_m"]:
                tested += 1
                try:
                    place = self._build_place_plan(
                        grasp, raw, dest, yaw_offset_deg=float(yaw_deg),
                        preplace_clearance_m=float(clearance),
                    )
                    pre = pose_spec_to_msg(place.preplace_pose, self.get_clock().now().to_msg())
                    release = pose_spec_to_msg(place.release_pose, self.get_clock().now().to_msg())
                    ompl, max_delta, l1 = self.moveit.plan_to_pose_from_current(pre)
                    start = self.moveit.current_robot_state()
                    at_pre = self.moveit.robot_state_after_trajectory(start, ompl.trajectory)
                    _, fraction = self.moveit.compute_cartesian_from_state(
                        at_pre, release, retime_for_execution=False
                    )
                    if fraction < min_fraction:
                        raise RuntimeError(
                            f"Cartesian preplace->release fraction={fraction:.3f} < {min_fraction:.3f}"
                        )
                    row = {
                        "yaw_offset_deg": float(yaw_deg),
                        "preplace_clearance_m": float(clearance),
                        "max_joint_delta_rad": float(max_delta),
                        "joint_l1_delta_rad": float(l1),
                        "planning_time_s": float(ompl.planning_time),
                        "release_cartesian_fraction": float(fraction),
                        "preplace_xyz": list(place.preplace_pose.position),
                        "release_xyz": list(place.release_pose.position),
                    }
                    feasible_this_yaw.append((float(l1), place, ompl, row))
                    self.get_logger().info(
                        f"Place candidate FEASIBLE: yaw={float(yaw_deg):+.1f}deg "
                        f"clearance={float(clearance)*1000:.0f}mm L1={float(l1):.3f} "
                        f"cart={float(fraction):.3f}"
                    )
                except Exception as exc:
                    row = {
                        "yaw_offset_deg": float(yaw_deg),
                        "preplace_clearance_m": float(clearance),
                        "error": str(exc),
                    }
                    failures.append(row)
                    self.get_logger().warn(
                        f"Place candidate rejected: yaw={float(yaw_deg):+.1f}deg "
                        f"clearance={float(clearance)*1000:.0f}mm: {exc}"
                    )

            if feasible_this_yaw:
                feasible_this_yaw.sort(key=lambda item: item[0])
                _, place, ompl, selected = feasible_this_yaw[0]
                selected["tested_candidates"] = tested
                selected["feasible_candidates_at_selected_yaw"] = len(feasible_this_yaw)
                selected["candidate_failures"] = failures
                return place, ompl, selected

        short = "; ".join(
            f"yaw={r['yaw_offset_deg']:+.0f},h={r['preplace_clearance_m']:.2f}: {r['error']}"
            for r in failures[:8]
        )
        raise RuntimeError(
            "No feasible place pose found after yaw/clearance search. " + short
        )

    def _save_place_plan(self, plan: PlacePlan, preview: Dict[str, Any]) -> None:
        payload = {
            "status": "ok",
            "created_at_unix_s": time.time(),
            **plan.to_dict(),
            "preview": preview,
        }
        atomic_write_json(str(self.p["latest_place_plan_json_path"]), payload)

    def _assert_pose_reached(self, target, pos_tol_m: float) -> Tuple[float, float]:
        actual = self.moveit.current_goal_link_pose(timeout_s=1.0)
        pos_err, ori_err = pose_error(actual, target)
        if pos_err > float(pos_tol_m):
            raise RuntimeError(f"Final position error={pos_err*1000:.1f}mm > {float(pos_tol_m)*1000:.1f}mm")
        if math.degrees(ori_err) > float(self.p["orientation_tolerance_deg"]):
            raise RuntimeError(
                f"Final orientation error={math.degrees(ori_err):.1f}deg > {float(self.p['orientation_tolerance_deg']):.1f}deg"
            )
        return pos_err, ori_err

    def _assert_world_ready(self) -> None:
        self.moveit.assert_current_state_valid()
        if not self.moveit.table_exists(str(self.p["table_object_id"])):
            raise RuntimeError("grasp_table is missing from MoveIt PlanningScene")

    def _srv_locate_destination(self, _, response):
        def op():
            query = str(self.get_parameter("destination_query").value).strip()
            obs = self.locator.locate(query, save_debug_image=True)
            atomic_write_json(str(self.p["latest_destination_json_path"]), obs.to_dict())
            self.destination = obs
            self.plan = None
            self._set_state(
                PlaceState.DESTINATION_LOCALIZED,
                "destination_localized",
                {
                    "query": obs.query,
                    "label": obs.label,
                    "confidence": obs.confidence,
                    "base_xy_m": list(obs.base_xy_m),
                    "debug_image_url": obs.debug_image_url,
                },
            )
            return (
                f"DESTINATION_LOCALIZED: {obs.label}, confidence={obs.confidence:.3f}, "
                f"base_xy=({obs.base_xy_m[0]:.4f},{obs.base_xy_m[1]:.4f})m, "
                f"debug={obs.debug_image_url}"
            )
        return self._service_guard(op, response)

    def _srv_prepare(self, _, response):
        def op():
            if self.state not in (PlaceState.DESTINATION_LOCALIZED, PlaceState.IDLE, PlaceState.ERROR):
                raise RuntimeError(f"prepare requires destination localized, current={self.state.value}")
            self.get_logger().info("place prepare[1/4]: checking MoveIt core services and current robot state...")
            self.moveit.wait_until_ready(5.0, require_execute=False)
            self._assert_world_ready()
            self.get_logger().info("place prepare[2/4]: loading recent PICKED evidence and grasp geometry...")
            grasp, raw, pick_row = self._load_grasp_context()
            self.get_logger().info("place prepare[3/4]: loading fresh destination and generating release geometry...")
            dest = self._load_destination()
            self.get_logger().info(
                "place prepare[4/4]: searching yaw/clearance candidates, previewing OMPL and Cartesian release..."
            )
            place, ompl, candidate = self._search_feasible_place_plan(grasp, raw, dest)
            fraction = float(candidate["release_cartesian_fraction"])
            preview = {
                "selected_yaw_offset_deg": float(candidate["yaw_offset_deg"]),
                "selected_preplace_clearance_m": float(candidate["preplace_clearance_m"]),
                "preplace_max_joint_delta_rad": float(candidate["max_joint_delta_rad"]),
                "preplace_joint_l1_delta_rad": float(candidate["joint_l1_delta_rad"]),
                "preplace_planning_time_s": float(candidate["planning_time_s"]),
                "release_cartesian_fraction": fraction,
                "tested_candidates": int(candidate["tested_candidates"]),
                "feasible_candidates_at_selected_yaw": int(candidate["feasible_candidates_at_selected_yaw"]),
                "candidate_failures": candidate["candidate_failures"],
                "pick_evidence_timestamp_unix_s": pick_row.get("timestamp_unix_s"),
            }
            self._save_place_plan(place, preview)
            self.destination = dest
            self.plan = place
            self.marker_pub.publish(
                make_place_markers(
                    str(self.p["base_frame"]),
                    self.get_clock().now().to_msg(),
                    place.destination_xy_m,
                    place.desired_grasp_center_xyz_m,
                    place.preplace_pose,
                    place.release_pose,
                    place.destination_label,
                )
            )
            self._set_state(PlaceState.PREPARED, "place_prepared", preview)
            return (
                f"PLACE_PREPARED: {place.source_label} -> {place.destination_label}; "
                f"destination_xy=({place.destination_xy_m[0]:.4f},{place.destination_xy_m[1]:.4f})m, "
                f"release_anchor_z={place.desired_grasp_center_xyz_m[2]:.4f}m, "
                f"yaw={preview['selected_yaw_offset_deg']:+.1f}deg, "
                f"clearance={preview['selected_preplace_clearance_m']*1000:.0f}mm, "
                f"cart={fraction:.3f}. "
                "No motion executed. Inspect /open_vocab/place/markers in RViz."
            )
        return self._service_guard(op, response)

    def _srv_move_preplace(self, _, response):
        def op():
            if self.state != PlaceState.PREPARED or self.plan is None:
                raise RuntimeError(f"move_preplace requires PREPARED, current={self.state.value}")
            if not bool(self.get_parameter("enable_place_arm_execution").value):
                raise RuntimeError("Place arm execution locked: enable_place_arm_execution=false")
            self.moveit.wait_until_ready(5.0, require_execute=True)
            self._assert_world_ready()
            target = pose_spec_to_msg(self.plan.preplace_pose, self.get_clock().now().to_msg())
            result, max_delta, l1 = self.moveit.plan_to_pose_from_current(target)
            self._set_state(
                PlaceState.MOVING_PREPLACE,
                "move_preplace_started",
                {"max_joint_delta_rad": max_delta, "joint_l1_delta_rad": l1},
            )
            self.moveit.execute_trajectory(result.trajectory)
            pos_err, ori_err = self._assert_pose_reached(
                target, float(self.p["preplace_position_tolerance_m"])
            )
            self._set_state(
                PlaceState.AT_PREPLACE,
                "at_preplace",
                {"position_error_m": pos_err, "orientation_error_rad": ori_err},
            )
            return (
                f"AT_PREPLACE: final error={pos_err*1000:.1f}mm / {math.degrees(ori_err):.1f}deg. "
                "Inspect bowl/box clearance before release."
            )
        return self._service_guard(op, response)

    def _srv_release_and_retreat(self, _, response):
        def op():
            if self.state != PlaceState.AT_PREPLACE or self.plan is None:
                raise RuntimeError(f"release_and_retreat requires AT_PREPLACE, current={self.state.value}")
            if not bool(self.get_parameter("enable_place_arm_execution").value):
                raise RuntimeError("Place arm execution locked")
            if not bool(self.get_parameter("enable_release_execution").value):
                raise RuntimeError("Release locked: enable_release_execution=false")
            self._assert_world_ready()
            self.gripper.wait_until_ready(3.0)

            release = pose_spec_to_msg(self.plan.release_pose, self.get_clock().now().to_msg())
            traj, fraction = self.moveit.compute_cartesian_from_current(release)
            self._set_state(PlaceState.DESCENDING, "place_descent_started", {"cartesian_fraction": fraction})
            self.moveit.execute_trajectory(traj)
            pos_err, ori_err = self._assert_pose_reached(
                release, float(self.p["release_position_tolerance_m"])
            )
            self._set_state(
                PlaceState.AT_RELEASE,
                "at_release",
                {"position_error_m": pos_err, "orientation_error_rad": ori_err},
            )

            self._set_state(PlaceState.RELEASING, "gripper_open_started", {})
            open_result = self.gripper.command(
                float(self.p["gripper_open_position"]),
                float(self.p["gripper_max_effort"]),
            )
            time.sleep(max(0.0, float(self.p["release_dwell_s"])))

            pre = pose_spec_to_msg(self.plan.preplace_pose, self.get_clock().now().to_msg())
            retreat, retreat_fraction = self.moveit.compute_cartesian_from_current(pre)
            self._set_state(
                PlaceState.RETREATING,
                "place_retreat_started",
                {"cartesian_fraction": retreat_fraction, "gripper": open_result.__dict__},
            )
            self.moveit.execute_trajectory(retreat)
            retreat_err, _ = self._assert_pose_reached(
                pre, float(self.p["preplace_position_tolerance_m"])
            )
            self._set_state(
                PlaceState.PLACED,
                "place_completed",
                {"retreat_position_error_m": retreat_err, "gripper": open_result.__dict__},
            )
            return (
                f"PLACED: released {self.plan.source_label} above {self.plan.destination_label} and "
                f"retreated {self.plan.preplace_clearance_m*1000:.0f}mm. "
                f"release_error={pos_err*1000:.1f}mm."
            )
        return self._service_guard(op, response)

    def _srv_verify(self, _, response):
        def op():
            if self.state not in (PlaceState.PLACED, PlaceState.VERIFIED):
                raise RuntimeError(f"verify requires PLACED, current={self.state.value}")
            if self.plan is None:
                raise RuntimeError("No place plan loaded")
            destination = self.locator.locate(self.plan.destination_query, save_debug_image=False)
            source_query = str(self.get_parameter("verify_source_query").value).strip() or self.plan.source_label
            source = self.locator.locate(source_query, save_debug_image=False)
            dx = source.base_xy_m[0] - destination.base_xy_m[0]
            dy = source.base_xy_m[1] - destination.base_xy_m[1]
            dist = math.hypot(dx, dy)
            radius = float(self.p["verify_acceptance_radius_m"])
            if dist > radius:
                raise RuntimeError(
                    f"Place verification failed: source/destination XY distance={dist:.3f}m > {radius:.3f}m"
                )
            self._set_state(
                PlaceState.VERIFIED,
                "place_verified",
                {
                    "source_xy_m": list(source.base_xy_m),
                    "destination_xy_m": list(destination.base_xy_m),
                    "xy_distance_m": dist,
                    "acceptance_radius_m": radius,
                },
            )
            return f"PLACE_VERIFIED: source/destination centroid distance={dist*1000:.1f}mm <= {radius*1000:.0f}mm."
        return self._service_guard(op, response)

    def _srv_abort(self, _, response):
        try:
            self.moveit.stop()
            self.gripper.cancel()
            self._set_state(PlaceState.ABORTED, "abort_requested", {})
            response.success = True
            response.message = "Software abort requested. Use physical E-stop for unsafe robot motion."
        except Exception as exc:
            response.success = False
            response.message = f"Abort error: {exc}"
        return response

    def _srv_reset(self, _, response):
        def op():
            if self.state in (
                PlaceState.MOVING_PREPLACE,
                PlaceState.DESCENDING,
                PlaceState.RELEASING,
                PlaceState.RETREATING,
            ):
                raise RuntimeError("Cannot reset during active motion; abort first")
            self.destination = None
            self.plan = None
            self.destination_file_mtime_ns = None
            self._set_state(PlaceState.IDLE, "reset", {})
            return "PlaceExecutor reset to IDLE."
        return self._service_guard(op, response)

    def _srv_status(self, _, response):
        response.success = True
        response.message = json.dumps(
            {
                "state": self.state.value,
                "destination_query": str(self.get_parameter("destination_query").value),
                "destination_label": self.destination.label if self.destination else None,
                "destination_xy_m": list(self.destination.base_xy_m) if self.destination else None,
                "place_arm_execution_enabled": bool(self.get_parameter("enable_place_arm_execution").value),
                "release_execution_enabled": bool(self.get_parameter("enable_release_execution").value),
                "destination_rim_height_m": float(self.get_parameter("destination_rim_height_m").value),
            },
            ensure_ascii=False,
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PlaceExecutorNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
