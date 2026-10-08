#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    PlanningScene,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetMotionPlan,
    GetPlanningScene,
    GetPositionIK,
)

from .geometry import build_goal_link_poses, shortest_angular_delta
from .grasp_server_client import GraspServerClient
from .markers import delete_all, make_candidate_markers

MOVEIT_SUCCESS = 1


def sec_to_duration(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


class GraspPlanNode(Node):
    """
    GraspNet V8 Top-K -> hardware checks -> table planning scene ->
    pregrasp IK -> OMPL -> collision-checked Cartesian approach.

    IMPORTANT: this node is PLAN-ONLY. It never sends ExecuteTrajectory and
    never actuates the gripper.
    """

    def __init__(self):
        super().__init__("open_vocab_grasp_plan")
        self.cb_group = ReentrantCallbackGroup()
        self._declare_parameters()

        gp = self.get_parameter
        self.latest_npz_path = gp("latest_npz_path").value
        self.server = GraspServerClient(
            gp("grasp_server_url").value,
            float(gp("grasp_server_timeout_s").value),
        )
        self.max_candidates = int(gp("max_candidates_to_test").value)
        self.base_frame = gp("base_frame").value
        self.group = gp("planning_group").value
        self.goal_link = gp("goal_link").value
        self.arm_joints = list(gp("arm_joint_names").value)
        self.planner_id = gp("planner_id").value
        self.planning_attempts = int(gp("planning_attempts").value)
        self.allowed_planning_time_s = float(gp("allowed_planning_time_s").value)
        self.vel_scale = float(gp("velocity_scaling").value)
        self.acc_scale = float(gp("acceleration_scaling").value)
        self.ik_timeout_s = float(gp("ik_timeout_s").value)

        self.R_grasp_to_goal = np.asarray(
            gp("grasp_to_goal_rotation").value, dtype=np.float64
        ).reshape(3, 3)
        if abs(np.linalg.det(self.R_grasp_to_goal) - 1.0) > 1e-3:
            raise RuntimeError("grasp_to_goal_rotation must be a proper rotation")

        # User-measured end_effector_link -> physical grasp center distance.
        self.goal_to_center = float(gp("goal_link_to_grasp_center_m").value)
        # Explicit task-space safety bias for tabletop grasps.  This translates
        # both pregrasp and grasp poses upward in base_link without falsifying
        # the physically measured EEF->grasp-center geometry.
        self.grasp_vertical_safety_offset = float(gp("grasp_vertical_safety_offset_m").value)
        self.pregrasp_distance = float(gp("pregrasp_distance_m").value)
        self.cartesian_step = float(gp("cartesian_max_step_m").value)
        self.cartesian_jump = float(gp("cartesian_jump_threshold").value)
        self.min_cartesian_fraction = float(gp("min_cartesian_fraction").value)
        self.min_width = float(gp("min_gripper_width_m").value)
        self.max_width = float(gp("max_gripper_width_m").value)
        self.width_margin = float(gp("width_open_margin_m").value)
        self.min_center_above_table = float(gp("min_grasp_center_above_table_m").value)
        self.max_single_joint_delta = float(gp("max_single_joint_delta_rad").value)
        self.max_joint_l1 = float(gp("max_joint_l1_delta_rad").value)
        self.motion_cost_weight = float(gp("motion_cost_weight").value)
        self.output_json_path = gp("output_json_path").value

        # Planning-scene table.  The collision box top can be shifted a few mm
        # below the measured physical plane to avoid numerical contact at z=table_z.
        self.add_table_collision = bool(gp("add_table_collision").value)
        self.table_object_id = str(gp("table_object_id").value)
        self.table_frame = str(gp("table_frame").value)
        self.table_center_x = float(gp("table_center_x_m").value)
        self.table_center_y = float(gp("table_center_y_m").value)
        self.table_top_z_cfg = float(gp("table_top_z_m").value)
        self.table_size_x = float(gp("table_size_x_m").value)
        self.table_size_y = float(gp("table_size_y_m").value)
        self.table_thickness = float(gp("table_thickness_m").value)
        self.table_surface_offset = float(gp("table_collision_surface_offset_m").value)
        self.table_use_npz_z = bool(gp("table_use_npz_z").value)

        self.latest_joint_state: Optional[JointState] = None
        self.create_subscription(
            JointState,
            gp("joint_states_topic").value,
            self._joint_cb,
            10,
            callback_group=self.cb_group,
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            gp("compute_ik_service").value,
            callback_group=self.cb_group,
        )
        self.plan_client = self.create_client(
            GetMotionPlan,
            gp("motion_plan_service").value,
            callback_group=self.cb_group,
        )
        self.cart_client = self.create_client(
            GetCartesianPath,
            gp("cartesian_path_service").value,
            callback_group=self.cb_group,
        )
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene,
            gp("apply_planning_scene_service").value,
            callback_group=self.cb_group,
        )
        self.get_scene_client = self.create_client(
            GetPlanningScene,
            gp("get_planning_scene_service").value,
            callback_group=self.cb_group,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray, gp("marker_topic").value, 10
        )
        self.create_service(
            Trigger,
            "/open_vocab/grasp/plan_latest",
            self._on_plan_latest,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger,
            "/open_vocab/grasp/validate_table",
            self._on_validate_table,
            callback_group=self.cb_group,
        )

        if bool(gp("execution_enabled").value):
            self.get_logger().error(
                "execution_enabled=true was requested, but this package is deliberately PLAN-ONLY."
            )

        self.get_logger().info(
            f"PLAN-ONLY grasp planner ready: group={self.group} goal_link={self.goal_link} "
            f"goal_link_to_grasp_center={self.goal_to_center:.3f}m"
        )
        if self.add_table_collision:
            self.get_logger().info(
                f"Table collision enabled: id={self.table_object_id} frame={self.table_frame} "
                f"size=({self.table_size_x:.3f},{self.table_size_y:.3f},{self.table_thickness:.3f})m"
            )
        else:
            self.get_logger().warn(
                "Table collision is DISABLED. Enable add_table_collision before real grasp execution."
            )
        self.get_logger().warn(
            "No real motion will be executed. Verify table, selected pregrasp and grasp in RViz first."
        )

    def _declare_parameters(self):
        d = self.declare_parameter
        d("latest_npz_path", "")
        d("grasp_server_url", "http://127.0.0.1:8091/api/v1/grasp_npz")
        d("grasp_server_timeout_s", 20.0)
        d("max_candidates_to_test", 10)

        d("base_frame", "base_link")
        d("planning_group", "manipulator")
        d("goal_link", "end_effector_link")
        d("compute_ik_service", "/compute_ik")
        d("motion_plan_service", "/plan_kinematic_path")
        d("cartesian_path_service", "/compute_cartesian_path")
        d("apply_planning_scene_service", "/apply_planning_scene")
        d("get_planning_scene_service", "/get_planning_scene")
        d("joint_states_topic", "/joint_states")
        d(
            "arm_joint_names",
            ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
        )

        d("planner_id", "RRTConnectkConfigDefault")
        d("planning_attempts", 3)
        d("allowed_planning_time_s", 3.0)
        d("velocity_scaling", 0.10)
        d("acceleration_scaling", 0.10)
        d("ik_timeout_s", 0.35)

        d(
            "grasp_to_goal_rotation",
            [0.0, 0.0, 1.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0],
        )
        d("goal_link_to_grasp_center_m", 0.145)
        d("grasp_vertical_safety_offset_m", 0.005)
        d("pregrasp_distance_m", 0.10)
        d("cartesian_max_step_m", 0.005)
        d("cartesian_jump_threshold", 0.0)
        d("min_cartesian_fraction", 0.98)

        d("min_gripper_width_m", 0.005)
        d("max_gripper_width_m", 0.085)
        d("width_open_margin_m", 0.008)
        d("min_grasp_center_above_table_m", 0.010)
        d("max_single_joint_delta_rad", 2.80)
        d("max_joint_l1_delta_rad", 8.50)
        d("motion_cost_weight", 0.025)
        d("execution_enabled", False)

        d("add_table_collision", True)
        d("table_object_id", "grasp_table")
        d("table_frame", "base_link")
        d("table_center_x_m", 0.50)
        d("table_center_y_m", 0.00)
        d("table_top_z_m", -0.031)
        d("table_size_x_m", 0.76)
        d("table_size_y_m", 0.80)
        d("table_thickness_m", 0.04)
        d("table_collision_surface_offset_m", 0.003)
        d("table_use_npz_z", True)

        d("output_json_path", "/tmp/latest_grasp_plan.json")
        d("marker_topic", "/open_vocab/grasp/markers")

    def _joint_cb(self, msg: JointState):
        self.latest_joint_state = copy.deepcopy(msg)

    def _wait_service(self, client, name: str, timeout_s: float = 5.0):
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"MoveIt service unavailable: {name}")

    def _wait_future(self, future, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS service call timed out")
            time.sleep(0.005)
        if future.exception() is not None:
            raise future.exception()
        return future.result()

    def _current_robot_state(self) -> RobotState:
        if self.latest_joint_state is None:
            raise RuntimeError("No /joint_states received yet")
        rs = RobotState()
        rs.joint_state = copy.deepcopy(self.latest_joint_state)
        rs.is_diff = False
        return rs

    @staticmethod
    def _joint_map(js: JointState) -> Dict[str, float]:
        return {n: float(p) for n, p in zip(js.name, js.position)}

    # ---------------------------------------------------------------------
    # Planning scene / table collision
    # ---------------------------------------------------------------------
    def _effective_table_top(self, measured_table_z: Optional[float]) -> float:
        physical_top = (
            float(measured_table_z)
            if self.table_use_npz_z and measured_table_z is not None
            else self.table_top_z_cfg
        )
        return physical_top - self.table_surface_offset

    def _make_table_collision_object(self, measured_table_z: Optional[float]) -> Tuple[CollisionObject, float, float]:
        effective_top = self._effective_table_top(measured_table_z)
        physical_top = (
            float(measured_table_z)
            if self.table_use_npz_z and measured_table_z is not None
            else self.table_top_z_cfg
        )

        obj = CollisionObject()
        obj.header.frame_id = self.table_frame
        obj.id = self.table_object_id

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [
            self.table_size_x,
            self.table_size_y,
            self.table_thickness,
        ]

        pose = Pose()
        pose.orientation.w = 1.0
        pose.position.x = self.table_center_x
        pose.position.y = self.table_center_y
        pose.position.z = effective_top - 0.5 * self.table_thickness

        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD
        return obj, physical_top, effective_top

    def _apply_table(self, measured_table_z: Optional[float]) -> Tuple[float, float]:
        if not self.add_table_collision:
            z = measured_table_z if measured_table_z is not None else self.table_top_z_cfg
            return float(z), float(z)

        self._wait_service(
            self.apply_scene_client,
            self.get_parameter("apply_planning_scene_service").value,
        )
        obj, physical_top, effective_top = self._make_table_collision_object(measured_table_z)

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [obj]

        req = ApplyPlanningScene.Request()
        req.scene = scene
        resp = self._wait_future(self.apply_scene_client.call_async(req), 3.0)
        if not resp.success:
            raise RuntimeError("MoveIt rejected ApplyPlanningScene for grasp_table")

        # Query it back.  This catches cases where a different move_group or
        # planning scene is being used than the one we think we are updating.
        verified = self._query_table_collision_object()
        if verified is None:
            raise RuntimeError(
                f"Applied table '{self.table_object_id}' but it was not found in /get_planning_scene"
            )

        self.get_logger().info(
            f"Planning-scene table verified: physical_top={physical_top:.4f}m "
            f"collision_top={effective_top:.4f}m center=({self.table_center_x:.3f},"
            f"{self.table_center_y:.3f},{effective_top - 0.5*self.table_thickness:.3f})m"
        )
        return physical_top, effective_top

    def _query_table_collision_object(self) -> Optional[CollisionObject]:
        self._wait_service(
            self.get_scene_client,
            self.get_parameter("get_planning_scene_service").value,
        )
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        resp = self._wait_future(self.get_scene_client.call_async(req), 3.0)
        for obj in resp.scene.world.collision_objects:
            if obj.id == self.table_object_id:
                return obj
        return None

    def _on_validate_table(self, _, response):
        try:
            measured_z = None
            p = Path(self.latest_npz_path)
            if self.table_use_npz_z and p.is_file():
                with np.load(p, allow_pickle=False) as data:
                    if "table_z" in data:
                        measured_z = float(data["table_z"])

            physical_top, collision_top = self._apply_table(measured_z)
            obj = self._query_table_collision_object()
            if obj is None or not obj.primitives or not obj.primitive_poses:
                raise RuntimeError("grasp_table not present after apply")

            dims = list(obj.primitives[0].dimensions)
            pose = obj.primitive_poses[0]
            response.success = True
            response.message = (
                f"TABLE VERIFIED in MoveIt: id={obj.id}, frame={obj.header.frame_id}, "
                f"physical_top={physical_top:.4f}m, collision_top={collision_top:.4f}m, "
                f"box_center=({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f})m, "
                f"size=({dims[0]:.3f},{dims[1]:.3f},{dims[2]:.3f})m. "
                f"Now confirm the box visually in RViz before planning."
            )
            return response
        except Exception as exc:
            self.get_logger().error(
                f"validate_table failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response

    # ---------------------------------------------------------------------
    # IK / planning
    # ---------------------------------------------------------------------
    def _compute_ik(self, pose, seed: RobotState) -> Optional[RobotState]:
        """Collision-aware IK for ROS 2 Humble / MoveIt 2.

        PositionIKRequest in Humble does NOT contain an ``attempts`` field.
        The kinematics plugin performs its search within ``timeout``.  We use
        the measured current robot state as the seed so redundant 7-DOF IK
        stays biased toward the current configuration; later joint-jump
        filters reject distant branches.
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.robot_state = copy.deepcopy(seed)
        req.ik_request.avoid_collisions = True
        req.ik_request.ik_link_name = self.goal_link
        req.ik_request.pose_stamped = copy.deepcopy(pose)
        req.ik_request.timeout = sec_to_duration(self.ik_timeout_s)

        try:
            resp = self._wait_future(
                self.ik_client.call_async(req),
                self.ik_timeout_s + 1.0,
            )
        except TimeoutError:
            self.get_logger().warn(
                f"IK service timeout after {self.ik_timeout_s:.3f}s "
                f"for link={self.goal_link}"
            )
            return None

        if resp is None:
            self.get_logger().warn("IK service returned no response")
            return None

        if resp.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().debug(
                f"IK failed: MoveItErrorCodes.val={resp.error_code.val} "
                f"link={self.goal_link}"
            )
            return None

        return resp.solution

    def _joint_motion_metrics(self, start: RobotState, goal: RobotState) -> Tuple[float, float]:
        s, g = self._joint_map(start.joint_state), self._joint_map(goal.joint_state)
        ds = [
            abs(shortest_angular_delta(s[n], g[n]))
            for n in self.arm_joints
            if n in s and n in g
        ]
        return (max(ds), sum(ds)) if ds else (float("inf"), float("inf"))

    def _goal_constraints(self, state: RobotState) -> Constraints:
        pos = self._joint_map(state.joint_state)
        c = Constraints()
        c.name = "pregrasp_joint_goal"
        for n in self.arm_joints:
            if n not in pos:
                continue
            jc = JointConstraint()
            jc.joint_name = n
            jc.position = pos[n]
            jc.tolerance_above = 1e-3
            jc.tolerance_below = 1e-3
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        if not c.joint_constraints:
            raise RuntimeError("IK solution does not contain configured arm joints")
        return c

    def _plan_to_state(self, start: RobotState, goal: RobotState):
        req = GetMotionPlan.Request()
        m = req.motion_plan_request
        m.group_name = self.group
        m.start_state = copy.deepcopy(start)
        m.goal_constraints = [self._goal_constraints(goal)]
        m.num_planning_attempts = self.planning_attempts
        m.allowed_planning_time = self.allowed_planning_time_s
        m.max_velocity_scaling_factor = self.vel_scale
        m.max_acceleration_scaling_factor = self.acc_scale
        m.planner_id = self.planner_id
        resp = self._wait_future(
            self.plan_client.call_async(req), self.allowed_planning_time_s + 2.0
        )
        r = resp.motion_plan_response
        if r.error_code.val != MOVEIT_SUCCESS or not r.trajectory.joint_trajectory.points:
            return None
        return r

    def _state_after_trajectory(self, current: RobotState, traj) -> RobotState:
        out = copy.deepcopy(current)
        mp = self._joint_map(out.joint_state)
        jt = traj.joint_trajectory
        last = jt.points[-1]
        for n, p in zip(jt.joint_names, last.positions):
            mp[n] = float(p)
        out.joint_state.position = [mp[n] for n in out.joint_state.name]
        out.is_diff = False
        return out

    def _cartesian_to_grasp(self, start: RobotState, grasp_pose):
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.start_state = copy.deepcopy(start)
        req.group_name = self.group
        req.link_name = self.goal_link
        req.waypoints = [grasp_pose.pose]
        req.max_step = self.cartesian_step
        req.jump_threshold = self.cartesian_jump
        req.avoid_collisions = True
        # IMPORTANT (ROS 2 Humble compatibility):
        # Some Humble moveit_msgs/GetCartesianPath definitions do NOT expose
        # max_velocity_scaling_factor / max_acceleration_scaling_factor on the
        # service request.  This planner is PLAN-ONLY and uses the Cartesian
        # result only as a collision/feasibility check, so do not write those
        # optional fields here.  Execution-time speed limiting is handled by
        # open_vocab_manipulation.
        return self._wait_future(self.cart_client.call_async(req), 5.0)

    def _hardware_ok(self, cand: dict, physical_table_z: Optional[float]):
        width = float(cand.get("width_m", 0.0))
        if not (self.min_width <= width <= self.max_width):
            return False, f"width={width:.4f}m outside hardware range"
        if physical_table_z is not None:
            z = float(np.asarray(cand["translation_base_grasp_m"])[2])
            clearance = z - physical_table_z
            if clearance < self.min_center_above_table:
                return False, (
                    f"grasp-center/table clearance={clearance:.4f}m "
                    f"< {self.min_center_above_table:.4f}m"
                )
        return True, "ok"

    def _source_npz_metadata(self) -> dict:
        """Read immutable metadata that binds a plan to the perception snapshot."""
        p = Path(self.latest_npz_path)
        if not p.is_file():
            return {}
        out = {
            "source_npz_path": str(p),
            "source_npz_mtime_ns": int(p.stat().st_mtime_ns),
        }
        try:
            with np.load(p, allow_pickle=False) as data:
                if "centroid_base" in data:
                    out["source_centroid_base_xyz_m"] = [
                        float(v) for v in np.asarray(data["centroid_base"]).reshape(-1)[:3]
                    ]
                if "stamp_ns" in data:
                    out["source_stamp_ns"] = int(np.asarray(data["stamp_ns"]).item())
                if "object_id" in data:
                    out["source_object_id"] = str(np.asarray(data["object_id"]).item())
                if "label" in data:
                    out["source_label"] = str(np.asarray(data["label"]).item())
        except Exception as exc:
            self.get_logger().warn(f"Could not read source NPZ metadata: {exc}")
        return out

    def _save_json(self, data: dict):
        p = Path(self.output_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    @staticmethod
    def _pose_dict(ps):
        return {
            "frame_id": ps.header.frame_id,
            "position": [
                ps.pose.position.x,
                ps.pose.position.y,
                ps.pose.position.z,
            ],
            "quaternion_xyzw": [
                ps.pose.orientation.x,
                ps.pose.orientation.y,
                ps.pose.orientation.z,
                ps.pose.orientation.w,
            ],
        }

    def _on_plan_latest(self, _, response):
        try:
            self._wait_service(self.ik_client, "/compute_ik")
            self._wait_service(self.plan_client, "/plan_kinematic_path")
            self._wait_service(self.cart_client, "/compute_cartesian_path")
            current = self._current_robot_state()

            self.get_logger().info(f"Uploading NPZ: {self.latest_npz_path}")
            srv = self.server.infer_npz(self.latest_npz_path)
            label = srv.get("label", "unknown")
            candidates = list(srv.get("candidates", []))[: self.max_candidates]
            measured_table_z = (
                float(srv["table_z_m"])
                if srv.get("table_z_m") is not None
                else None
            )
            if not candidates:
                response.success = False
                response.message = f"No final GraspNet candidates for {label}"
                return response

            # Critical: make the world collision object part of the same MoveIt
            # PlanningScene used by IK, OMPL, and GetCartesianPath.
            physical_table_z, collision_table_top = self._apply_table(measured_table_z)

            now = self.get_clock().now().to_msg()
            self.marker_pub.publish(delete_all(self.base_frame, now))
            rows = []

            for i, cand in enumerate(candidates):
                row = {
                    "candidate": cand,
                    "rank": int(cand.get("rank", i)),
                    "feasible": False,
                    "reason": "",
                    "selection_score": None,
                }
                pre, grasp, approach = build_goal_link_poses(
                    cand,
                    self.base_frame,
                    now,
                    self.R_grasp_to_goal,
                    self.goal_to_center,
                    self.pregrasp_distance,
                )
                # Tabletop safety bias: closing a 2F-85 changes the finger geometry
                # and can reduce fingertip/table clearance even when the open-gripper
                # dry-run is collision-free.  Translate the whole approach upward in
                # base_link by a small explicit amount rather than changing the
                # calibrated EEF->grasp-center distance.
                if self.grasp_vertical_safety_offset != 0.0:
                    pre.pose.position.z += self.grasp_vertical_safety_offset
                    grasp.pose.position.z += self.grasp_vertical_safety_offset
                row.update(
                    pregrasp_pose=pre,
                    grasp_pose=grasp,
                    approach=approach.tolist(),
                )

                ok, why = self._hardware_ok(cand, physical_table_z)
                if not ok:
                    row["reason"] = why
                    rows.append(row)
                    self.get_logger().warn(f"Candidate {i}: {why}")
                    continue

                ik = self._compute_ik(pre, current)
                if ik is None:
                    row["reason"] = "pregrasp IK failed/collides"
                    rows.append(row)
                    self.get_logger().warn(
                        f"Candidate {i}: pregrasp IK failed/collides"
                    )
                    continue

                max_d, l1 = self._joint_motion_metrics(current, ik)
                row["max_joint_delta_rad"] = max_d
                row["joint_l1_delta_rad"] = l1
                if max_d > self.max_single_joint_delta:
                    row["reason"] = (
                        f"joint jump {max_d:.3f}rad > {self.max_single_joint_delta:.3f}"
                    )
                    rows.append(row)
                    self.get_logger().warn(f"Candidate {i}: {row['reason']}")
                    continue
                if l1 > self.max_joint_l1:
                    row["reason"] = (
                        f"joint L1 {l1:.3f}rad > {self.max_joint_l1:.3f}"
                    )
                    rows.append(row)
                    self.get_logger().warn(f"Candidate {i}: {row['reason']}")
                    continue

                plan = self._plan_to_state(current, ik)
                if plan is None:
                    row["reason"] = "OMPL current->pregrasp failed/collides"
                    rows.append(row)
                    self.get_logger().warn(f"Candidate {i}: OMPL failed/collides")
                    continue

                at_pre = self._state_after_trajectory(current, plan.trajectory)
                cart = self._cartesian_to_grasp(at_pre, grasp)
                fraction = float(cart.fraction)
                row["cartesian_fraction"] = fraction
                if fraction < self.min_cartesian_fraction:
                    row["reason"] = (
                        f"Cartesian fraction={fraction:.3f} < "
                        f"{self.min_cartesian_fraction:.3f}; likely collision/unreachable"
                    )
                    rows.append(row)
                    self.get_logger().warn(f"Candidate {i}: {row['reason']}")
                    continue

                model_score = float(cand.get("model_score", 0.0))
                row["feasible"] = True
                row["reason"] = "ok"
                row["selection_score"] = (
                    model_score - self.motion_cost_weight * l1
                )
                row["planning_time_s"] = float(plan.planning_time)
                row["suggested_open_width_m"] = min(
                    self.max_width,
                    float(cand["width_m"]) + self.width_margin,
                )
                rows.append(row)
                self.get_logger().info(
                    f"Candidate {i} FEASIBLE: G={model_score:.3f} L1={l1:.3f} "
                    f"cart={fraction:.3f} select={row['selection_score']:.3f}"
                )

            feasible = [i for i, r in enumerate(rows) if r["feasible"]]
            selected_idx = (
                max(feasible, key=lambda i: rows[i]["selection_score"])
                if feasible
                else None
            )
            self.marker_pub.publish(
                make_candidate_markers(
                    self.base_frame, now, rows, selected_idx
                )
            )

            json_rows = []
            for r in rows:
                json_rows.append(
                    {
                        "rank": r["rank"],
                        "candidate": r["candidate"],
                        "feasible": r["feasible"],
                        "reason": r["reason"],
                        "selection_score": r.get("selection_score"),
                        "max_joint_delta_rad": r.get("max_joint_delta_rad"),
                        "joint_l1_delta_rad": r.get("joint_l1_delta_rad"),
                        "cartesian_fraction": r.get("cartesian_fraction"),
                        "planning_time_s": r.get("planning_time_s"),
                        "suggested_open_width_m": r.get("suggested_open_width_m"),
                        "approach_axis_base": r.get("approach"),
                        "pregrasp_goal_link_pose": self._pose_dict(r["pregrasp_pose"]),
                        "grasp_goal_link_pose": self._pose_dict(r["grasp_pose"]),
                    }
                )

            source_meta = self._source_npz_metadata()
            out = {
                "status": "ok" if selected_idx is not None else "no_feasible_candidate",
                "created_at_unix_s": time.time(),
                "created_at_ros_time_ns": int(self.get_clock().now().nanoseconds),
                **source_meta,
                "label": label,
                "object_id": srv.get("object_id", ""),
                "base_frame": self.base_frame,
                "planning_group": self.group,
                "goal_link": self.goal_link,
                "goal_link_to_grasp_center_m": self.goal_to_center,
                "plan_only": True,
                "table_collision": {
                    "enabled": self.add_table_collision,
                    "object_id": self.table_object_id,
                    "physical_top_z_m": physical_table_z,
                    "collision_top_z_m": collision_table_top,
                    "surface_offset_m": self.table_surface_offset,
                },
                "tested_candidates": len(rows),
                "feasible_candidates": len(feasible),
                "selected": (
                    json_rows[selected_idx]
                    if selected_idx is not None
                    else None
                ),
                "candidates": json_rows,
            }
            self._save_json(out)

            if selected_idx is None:
                response.success = False
                response.message = (
                    f"{label}: {len(candidates)} GraspNet candidates, none passed "
                    f"IK+OMPL+Cartesian with table collision. See {self.output_json_path}"
                )
                return response

            sel = rows[selected_idx]
            cand = sel["candidate"]
            response.success = True
            response.message = (
                f"PLAN-ONLY OK: {label}, selected rank={cand.get('rank', selected_idx)}, "
                f"G={float(cand.get('model_score', 0)):.3f}, "
                f"width={float(cand.get('width_m', 0))*1000:.1f}mm, "
                f"joint_L1={sel['joint_l1_delta_rad']:.3f}rad, "
                f"cart={sel['cartesian_fraction']:.3f}, "
                f"table_collision={'ON' if self.add_table_collision else 'OFF'}, "
                f"EEF->grasp_center={self.goal_to_center*1000:.0f}mm. "
                f"NO MOTION EXECUTED. Saved={self.output_json_path}"
            )
            return response

        except Exception as exc:
            self.get_logger().error(
                f"plan_latest failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
            return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspPlanNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
