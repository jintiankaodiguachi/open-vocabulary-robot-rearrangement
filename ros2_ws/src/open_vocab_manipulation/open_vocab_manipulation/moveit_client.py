from __future__ import annotations

import copy
import math
import threading
import time
from typing import Dict, Optional, Tuple

from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, PlanningSceneComponents, RobotState
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan, GetPlanningScene, GetPositionIK, GetStateValidity
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .geometry import transform_to_pose_stamped

MOVEIT_SUCCESS = 1


def _sec_to_duration_msg(seconds: float):
    from builtin_interfaces.msg import Duration as DurationMsg

    seconds = max(0.0, float(seconds))
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return DurationMsg(sec=sec, nanosec=nanosec)


def _duration_msg_to_sec(msg) -> float:
    return float(msg.sec) + float(msg.nanosec) * 1e-9


def _retime_trajectory_slower_in_place(trajectory, velocity_scale: float, acceleration_scale: float) -> None:
    """Conservatively slow an already time-parameterized JointTrajectory.

    Older ROS 2 Humble moveit_msgs/GetCartesianPath request definitions do not
    contain velocity/acceleration scaling fields.  In that case MoveIt may still
    return a time-parameterized trajectory at its default limits.  We only slow
    that trajectory down; we never speed it up.

    For a time stretch factor k:
        t' = k t, v' = v/k, a' = a/k^2
    Choosing k >= 1/v_scale and k >= 1/sqrt(a_scale) enforces both requested
    scaling bounds relative to the returned trajectory.
    """
    pts = trajectory.joint_trajectory.points
    if not pts:
        raise RuntimeError("Cannot retime an empty Cartesian trajectory")

    v_scale = min(1.0, max(1e-3, float(velocity_scale)))
    a_scale = min(1.0, max(1e-3, float(acceleration_scale)))
    stretch = max(1.0 / v_scale, 1.0 / math.sqrt(a_scale))

    # A trajectory with every time_from_start == 0 is not safe to send to a
    # FollowJointTrajectory-style controller.  Refuse rather than inventing
    # timing from geometric waypoints in Python.
    times = [_duration_msg_to_sec(pt.time_from_start) for pt in pts]
    if len(pts) > 1 and max(times) <= 0.0:
        raise RuntimeError(
            "Cartesian trajectory has no time parameterization in this Humble "
            "MoveIt build; refusing execution. Add a MoveIt-side TOTG/IPTP "
            "retiming service or use a compatible MoveIt build before enabling "
            "Cartesian execution."
        )

    for pt, t in zip(pts, times):
        pt.time_from_start = _sec_to_duration_msg(t * stretch)
        if pt.velocities:
            pt.velocities = [float(v) / stretch for v in pt.velocities]
        if pt.accelerations:
            pt.accelerations = [float(a) / (stretch * stretch) for a in pt.accelerations]


class MoveItExecutionClient:
    def __init__(self, node, callback_group, params: Dict[str, object]):
        self.node = node
        self.cb_group = callback_group
        self.group = str(params["planning_group"])
        self.base_frame = str(params["base_frame"])
        self.goal_link = str(params["goal_link"])
        self.arm_joints = list(params["arm_joint_names"])
        self.planner_id = str(params["planner_id"])
        self.planning_attempts = int(params["planning_attempts"])
        self.allowed_planning_time_s = float(params["allowed_planning_time_s"])
        self.ik_timeout_s = float(params["ik_timeout_s"])
        self.vel_scale = float(params["velocity_scaling"])
        self.acc_scale = float(params["acceleration_scaling"])
        self.cartesian_vel_scale = float(params["cartesian_velocity_scaling"])
        self.cartesian_acc_scale = float(params["cartesian_acceleration_scaling"])
        self.cartesian_step = float(params["cartesian_max_step_m"])
        self.cartesian_jump = float(params["cartesian_jump_threshold"])
        self.min_cartesian_fraction = float(params["min_cartesian_fraction"])
        self.max_single_joint_delta = float(params["max_single_joint_delta_rad"])
        self.max_joint_l1 = float(params["max_joint_l1_delta_rad"])
        self.max_joint_state_age_s = float(params["max_joint_state_age_s"])
        self.execution_timeout_s = float(params["execution_timeout_s"])
        self.require_state_validity = bool(params["require_state_validity_service"])
        self.table_object_id = str(params["table_object_id"])

        self._latest_joint_state: Optional[JointState] = None
        self._latest_joint_receive_monotonic = 0.0
        self._joint_lock = threading.Lock()
        self._active_goal_handle = None
        self._action_lock = threading.Lock()

        self.node.create_subscription(
            JointState,
            str(params["joint_states_topic"]),
            self._joint_cb,
            20,
            callback_group=callback_group,
        )
        self.ik_client = self.node.create_client(
            GetPositionIK, str(params["compute_ik_service"]), callback_group=callback_group
        )
        self.plan_client = self.node.create_client(
            GetMotionPlan, str(params["motion_plan_service"]), callback_group=callback_group
        )
        self.cart_client = self.node.create_client(
            GetCartesianPath, str(params["cartesian_path_service"]), callback_group=callback_group
        )
        self.scene_client = self.node.create_client(
            GetPlanningScene, str(params["get_planning_scene_service"]), callback_group=callback_group
        )
        self.validity_client = self.node.create_client(
            GetStateValidity, str(params["state_validity_service"]), callback_group=callback_group
        )
        self.execute_action = ActionClient(
            self.node,
            ExecuteTrajectory,
            str(params["execute_trajectory_action"]),
            callback_group=callback_group,
        )
        self.trajectory_event_pub = self.node.create_publisher(
            String, str(params["trajectory_execution_event_topic"]), 10
        )
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=False)

    def _joint_cb(self, msg: JointState) -> None:
        with self._joint_lock:
            self._latest_joint_state = copy.deepcopy(msg)
            self._latest_joint_receive_monotonic = time.monotonic()

    @staticmethod
    def _wait_future(future, timeout_s: float):
        deadline = time.monotonic() + float(timeout_s)
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS async operation timed out")
            time.sleep(0.005)
        if future.exception() is not None:
            raise future.exception()
        return future.result()

    @staticmethod
    def _wait_service(client, name: str, timeout_s: float = 5.0) -> None:
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"Required MoveIt service unavailable: {name}")

    def wait_until_ready(self, timeout_s: float = 5.0, require_execute: bool = True) -> None:
        self._wait_service(self.ik_client, "compute_ik", timeout_s)
        self._wait_service(self.plan_client, "plan_kinematic_path", timeout_s)
        self._wait_service(self.cart_client, "compute_cartesian_path", timeout_s)
        self._wait_service(self.scene_client, "get_planning_scene", timeout_s)
        if self.require_state_validity:
            self._wait_service(self.validity_client, "check_state_validity", timeout_s)
        if require_execute and not self.execute_action.wait_for_server(timeout_sec=timeout_s):
            raise RuntimeError("MoveIt /execute_trajectory action server unavailable")

    def current_robot_state(self) -> RobotState:
        with self._joint_lock:
            js = copy.deepcopy(self._latest_joint_state)
            age = time.monotonic() - self._latest_joint_receive_monotonic
        if js is None:
            raise RuntimeError("No /joint_states received yet")
        if age > self.max_joint_state_age_s:
            raise RuntimeError(
                f"/joint_states stale: age={age:.3f}s > {self.max_joint_state_age_s:.3f}s"
            )
        missing = [n for n in self.arm_joints if n not in js.name]
        if missing:
            raise RuntimeError(f"/joint_states missing configured arm joints: {missing}")
        state = RobotState()
        state.joint_state = js
        state.is_diff = False
        return state

    @staticmethod
    def _joint_map(js: JointState) -> Dict[str, float]:
        return {n: float(p) for n, p in zip(js.name, js.position)}

    @staticmethod
    def _shortest_angular_delta(a: float, b: float) -> float:
        return (float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi

    def joint_motion_metrics(self, start: RobotState, goal: RobotState) -> Tuple[float, float]:
        s = self._joint_map(start.joint_state)
        g = self._joint_map(goal.joint_state)
        deltas = [
            abs(self._shortest_angular_delta(s[n], g[n]))
            for n in self.arm_joints
            if n in s and n in g
        ]
        if not deltas:
            return float("inf"), float("inf")
        return max(deltas), sum(deltas)

    def assert_current_state_valid(self) -> None:
        if not self.require_state_validity:
            return
        req = GetStateValidity.Request()
        req.robot_state = self.current_robot_state()
        req.group_name = self.group
        resp = self._wait_future(self.validity_client.call_async(req), 3.0)
        if not bool(resp.valid):
            contacts = [f"{c.contact_body_1}<->{c.contact_body_2}" for c in resp.contacts[:8]]
            raise RuntimeError(
                "Current robot state is in collision/invalid according to MoveIt"
                + (f": {contacts}" if contacts else "")
            )

    def table_exists(self, expected_id: Optional[str] = None) -> bool:
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        resp = self._wait_future(self.scene_client.call_async(req), 3.0)
        object_id = expected_id or self.table_object_id
        return any(obj.id == object_id for obj in resp.scene.world.collision_objects)

    def compute_ik(self, pose: PoseStamped, seed: RobotState) -> RobotState:
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.robot_state = copy.deepcopy(seed)
        req.ik_request.avoid_collisions = True
        req.ik_request.ik_link_name = self.goal_link
        req.ik_request.pose_stamped = copy.deepcopy(pose)
        req.ik_request.timeout = _sec_to_duration_msg(self.ik_timeout_s)
        resp = self._wait_future(self.ik_client.call_async(req), self.ik_timeout_s + 1.0)
        if resp.error_code.val != MOVEIT_SUCCESS:
            raise RuntimeError(f"Collision-aware IK failed, MoveIt code={resp.error_code.val}")
        return resp.solution

    def _goal_constraints(self, state: RobotState) -> Constraints:
        pos = self._joint_map(state.joint_state)
        constraints = Constraints()
        constraints.name = "execution_joint_goal"
        for name in self.arm_joints:
            if name not in pos:
                continue
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos[name]
            jc.tolerance_above = 1e-3
            jc.tolerance_below = 1e-3
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        if not constraints.joint_constraints:
            raise RuntimeError("IK solution contains none of the configured arm joints")
        return constraints

    def plan_to_pose_from_current(self, target: PoseStamped):
        start = self.current_robot_state()
        goal_state = self.compute_ik(target, start)
        max_delta, l1 = self.joint_motion_metrics(start, goal_state)
        if max_delta > self.max_single_joint_delta:
            raise RuntimeError(
                f"Joint jump guard failed: max={max_delta:.3f}rad > {self.max_single_joint_delta:.3f}rad"
            )
        if l1 > self.max_joint_l1:
            raise RuntimeError(
                f"Joint L1 guard failed: L1={l1:.3f}rad > {self.max_joint_l1:.3f}rad"
            )

        req = GetMotionPlan.Request()
        m = req.motion_plan_request
        m.group_name = self.group
        m.start_state = copy.deepcopy(start)
        m.goal_constraints = [self._goal_constraints(goal_state)]
        m.num_planning_attempts = self.planning_attempts
        m.allowed_planning_time = self.allowed_planning_time_s
        m.max_velocity_scaling_factor = self.vel_scale
        m.max_acceleration_scaling_factor = self.acc_scale
        m.planner_id = self.planner_id
        resp = self._wait_future(
            self.plan_client.call_async(req), self.allowed_planning_time_s + 2.0
        )
        result = resp.motion_plan_response
        if result.error_code.val != MOVEIT_SUCCESS:
            raise RuntimeError(f"OMPL plan failed, MoveIt code={result.error_code.val}")
        if not result.trajectory.joint_trajectory.points:
            raise RuntimeError("OMPL returned an empty trajectory")
        return result, max_delta, l1

    def robot_state_after_trajectory(self, start: RobotState, trajectory) -> RobotState:
        out = copy.deepcopy(start)
        if not trajectory.joint_trajectory.points:
            raise RuntimeError("Cannot derive robot state from empty trajectory")
        mapping = self._joint_map(out.joint_state)
        jt = trajectory.joint_trajectory
        last = jt.points[-1]
        for name, position in zip(jt.joint_names, last.positions):
            mapping[name] = float(position)
        out.joint_state.position = [mapping[n] for n in out.joint_state.name]
        out.is_diff = False
        return out

    def compute_cartesian_from_state(
        self, start: RobotState, target: PoseStamped, *, retime_for_execution: bool
    ):
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.start_state = copy.deepcopy(start)
        req.group_name = self.group
        req.link_name = self.goal_link
        req.waypoints = [copy.deepcopy(target.pose)]
        req.max_step = self.cartesian_step
        req.jump_threshold = self.cartesian_jump
        req.avoid_collisions = True

        # moveit_msgs/GetCartesianPath differs across Humble package revisions.
        # Newer definitions expose explicit scaling fields; older Humble builds
        # (including the user's current environment) do not.
        request_has_scaling = hasattr(req, "max_velocity_scaling_factor")
        if request_has_scaling:
            req.max_velocity_scaling_factor = self.cartesian_vel_scale
            req.max_acceleration_scaling_factor = self.cartesian_acc_scale

        resp = self._wait_future(self.cart_client.call_async(req), 5.0)
        if resp.error_code.val != MOVEIT_SUCCESS:
            raise RuntimeError(f"Cartesian planning failed, MoveIt code={resp.error_code.val}")
        if float(resp.fraction) < self.min_cartesian_fraction:
            raise RuntimeError(
                f"Cartesian fraction={resp.fraction:.4f} < required {self.min_cartesian_fraction:.4f}"
            )
        if not resp.solution.joint_trajectory.points:
            raise RuntimeError("Cartesian planner returned empty trajectory")

        # On older Humble service definitions the request cannot carry speed
        # scaling.  Only execution trajectories need conservative retiming;
        # feasibility previews can stay untouched.
        if retime_for_execution and not request_has_scaling:
            _retime_trajectory_slower_in_place(
                resp.solution,
                self.cartesian_vel_scale,
                self.cartesian_acc_scale,
            )
        return resp.solution, float(resp.fraction)

    def compute_cartesian_from_current(self, target: PoseStamped):
        return self.compute_cartesian_from_state(
            self.current_robot_state(), target, retime_for_execution=True
        )

    def execute_trajectory(self, trajectory) -> None:
        # ROS 2 Humble moveit_msgs/action/ExecuteTrajectory.Goal contains only
        # `trajectory`. Controller selection is handled by MoveIt's
        # TrajectoryExecutionManager using the configured MoveIt controllers.
        if not self.execute_action.wait_for_server(timeout_sec=3.0):
            raise RuntimeError("MoveIt execute_trajectory action server unavailable")

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = copy.deepcopy(trajectory)
        send_future = self.execute_action.send_goal_async(goal)
        goal_handle = self._wait_future(send_future, 3.0)
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("MoveIt rejected ExecuteTrajectory goal")

        with self._action_lock:
            self._active_goal_handle = goal_handle
        try:
            result_wrap = self._wait_future(
                goal_handle.get_result_async(), self.execution_timeout_s
            )
        finally:
            with self._action_lock:
                self._active_goal_handle = None

        if result_wrap is None:
            raise RuntimeError("ExecuteTrajectory returned no result")
        result = result_wrap.result
        if result.error_code.val != MOVEIT_SUCCESS:
            raise RuntimeError(
                f"ExecuteTrajectory failed: action_status={result_wrap.status}, "
                f"MoveIt code={result.error_code.val}"
            )

    def stop(self) -> None:
        msg = String()
        msg.data = "stop"
        self.trajectory_event_pub.publish(msg)
        with self._action_lock:
            handle = self._active_goal_handle
        if handle is not None:
            try:
                self._wait_future(handle.cancel_goal_async(), 2.0)
            except Exception as exc:
                self.node.get_logger().error(f"Failed to cancel active trajectory: {exc}")

    def current_goal_link_pose(self, timeout_s: float = 1.0) -> PoseStamped:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.goal_link,
                Time(),
                timeout=Duration(seconds=float(timeout_s)),
            )
        except TransformException as exc:
            raise RuntimeError(
                f"TF unavailable: {self.base_frame} -> {self.goal_link}: {exc}"
            ) from exc
        return transform_to_pose_stamped(tf)
