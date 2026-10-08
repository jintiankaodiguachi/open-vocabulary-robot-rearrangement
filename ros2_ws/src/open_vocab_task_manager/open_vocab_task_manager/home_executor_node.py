#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    PlanningScene,
    PlanningSceneComponents,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan, GetPlanningScene, GetStateValidity
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .journal import atomic_write_json

MOVEIT_SUCCESS = 1


class HomeExecutorNode(Node):
    """Deterministic joint-space home/initial-point executor.

    The home pose is a *joint-space* configuration, not a Cartesian pose. This is
    intentional: returning to a fixed Cartesian TCP pose through IK can select a
    different 7-DoF solution and cause unnecessary wrist/arm rotations. A stored
    joint configuration gives repeatable task start/end posture.

    Home is configured either by:
      1) a persisted JSON captured with /open_vocab/home/capture_current, or
      2) home_joint_positions in home.yaml as a fallback.

    Motion is always planned by MoveIt from the *current* RobotState and then
    executed through /execute_trajectory. No cached trajectory is replayed.
    """

    def __init__(self) -> None:
        super().__init__("open_vocab_home_executor")
        self.cb_group = ReentrantCallbackGroup()
        self._declare_parameters()
        self.p = {name: self.get_parameter(name).value for name in self._parameters.keys()}

        self.arm_joint_names = [str(x) for x in self.p["arm_joint_names"]]
        if len(self.arm_joint_names) != 7:
            self.get_logger().warn(
                f"Configured arm_joint_names has {len(self.arm_joint_names)} entries; expected 7 for Kinova Gen3."
            )

        self._joint_lock = threading.Lock()
        self._latest_joint_state: Optional[JointState] = None
        self._latest_joint_receive_monotonic = 0.0
        self._motion_lock = threading.Lock()
        self._active_goal_handle = None
        self._active_goal_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = "IDLE"
        self._message = "HomeExecutor ready"

        self.create_subscription(
            JointState,
            str(self.p["joint_states_topic"]),
            self._joint_cb,
            20,
            callback_group=self.cb_group,
        )

        self.plan_client = self.create_client(
            GetMotionPlan,
            str(self.p["motion_plan_service"]),
            callback_group=self.cb_group,
        )
        self.scene_client = self.create_client(
            GetPlanningScene,
            str(self.p["get_planning_scene_service"]),
            callback_group=self.cb_group,
        )
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene,
            str(self.p["apply_planning_scene_service"]),
            callback_group=self.cb_group,
        )
        self.validity_client = self.create_client(
            GetStateValidity,
            str(self.p["state_validity_service"]),
            callback_group=self.cb_group,
        )
        self.execute_action = ActionClient(
            self,
            ExecuteTrajectory,
            str(self.p["execute_trajectory_action"]),
            callback_group=self.cb_group,
        )
        self.trajectory_event_pub = self.create_publisher(
            String,
            str(self.p["trajectory_execution_event_topic"]),
            10,
        )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.status_pub = self.create_publisher(String, str(self.p["status_topic"]), qos)

        self.capture_srv = self.create_service(
            Trigger,
            str(self.p["capture_service"]),
            self._on_capture,
            callback_group=self.cb_group,
        )
        self.go_srv = self.create_service(
            Trigger,
            str(self.p["go_service"]),
            self._on_go,
            callback_group=self.cb_group,
        )
        self.abort_srv = self.create_service(
            Trigger,
            str(self.p["abort_service"]),
            self._on_abort,
            callback_group=self.cb_group,
        )
        self.status_srv = self.create_service(
            Trigger,
            str(self.p["status_service"]),
            self._on_status,
            callback_group=self.cb_group,
        )

        configured, source, _, _ = self._load_home_configuration(raise_if_missing=False)
        self._publish_status()
        self.get_logger().info(
            "HomeExecutor ready: "
            f"configured={configured} source={source or 'none'} "
            f"execution_enabled={bool(self.p['enable_home_execution'])}"
        )
        if not configured:
            self.get_logger().warn(
                "No HOME pose configured. Put the robot in the desired safe initial pose and call "
                f"{self.p['capture_service']} once."
            )
        if not bool(self.p["enable_home_execution"]):
            self.get_logger().warn("HOME EXECUTION IS LOCKED (enable_home_execution=false).")

    def _declare_parameters(self) -> None:
        d = self.declare_parameter
        d("planning_group", "manipulator")
        d("joint_states_topic", "/joint_states")
        d("arm_joint_names", [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"
        ])

        d("motion_plan_service", "/plan_kinematic_path")
        d("get_planning_scene_service", "/get_planning_scene")
        d("state_validity_service", "/check_state_validity")
        d("execute_trajectory_action", "/execute_trajectory")
        d("trajectory_execution_event_topic", "/trajectory_execution_event")

        d("planner_id", "RRTConnectkConfigDefault")
        d("planning_attempts", 5)
        d("allowed_planning_time_s", 5.0)
        d("velocity_scaling", 0.08)
        d("acceleration_scaling", 0.06)
        d("max_joint_state_age_s", 0.50)
        d("execution_timeout_s", 60.0)
        d("goal_joint_tolerance_rad", 0.008)
        d("already_home_tolerance_rad", 0.035)
        d("final_home_tolerance_rad", 0.045)

        d("require_state_validity_service", True)

        # HOME must not depend on grasp_plan_latest() to populate the table.
        # If the table is absent, HomeExecutor can insert/update the static table
        # itself before planning the return-to-HOME trajectory.
        d("ensure_table_collision_object", True)
        d("require_table_collision_object", True)
        d("apply_planning_scene_service", "/apply_planning_scene")
        d("table_object_id", "grasp_table")
        d("table_frame", "base_link")
        d("table_center_x_m", 0.500)
        d("table_center_y_m", 0.000)
        d("table_physical_top_z_m", -0.031)
        d("table_collision_surface_offset_m", -0.002)
        d("table_size_x_m", 0.760)
        d("table_size_y_m", 0.800)
        d("table_size_z_m", 0.040)

        d("home_pose_path", "/opt/open-vocabulary-robot-rearrangement/perception/system_state/home_pose.json")
        # Empty list means: use persisted capture file. A 7-element list can be
        # provided as a reproducible config fallback.
        # Humble rclpy cannot reliably initialize an untyped empty array parameter.
        # A one-element DOUBLE_ARRAY sentinel keeps the parameter initialized while
        # still meaning "not configured" because only exactly 7 entries are accepted.
        d("home_joint_positions", [0.0])

        d("enable_home_execution", False)
        d("capture_service", "/open_vocab/home/capture_current")
        d("go_service", "/open_vocab/home/go")
        d("abort_service", "/open_vocab/home/abort")
        d("status_service", "/open_vocab/home/status")
        d("status_topic", "/open_vocab/home/status_topic")

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
        exc = future.exception()
        if exc is not None:
            raise exc
        return future.result()

    def _current_joint_state(self) -> JointState:
        with self._joint_lock:
            js = copy.deepcopy(self._latest_joint_state)
            age = time.monotonic() - self._latest_joint_receive_monotonic
        if js is None:
            raise RuntimeError("No /joint_states received yet")
        if age > float(self.p["max_joint_state_age_s"]):
            raise RuntimeError(
                f"/joint_states stale: age={age:.3f}s > {float(self.p['max_joint_state_age_s']):.3f}s"
            )
        missing = [name for name in self.arm_joint_names if name not in js.name]
        if missing:
            raise RuntimeError(f"/joint_states missing configured arm joints: {missing}")
        return js

    @staticmethod
    def _joint_map(js: JointState) -> Dict[str, float]:
        return {name: float(pos) for name, pos in zip(js.name, js.position)}

    @staticmethod
    def _shortest_delta(a: float, b: float) -> float:
        return (float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi

    def _joint_errors(self, current: JointState, target: Dict[str, float]) -> Dict[str, float]:
        now = self._joint_map(current)
        return {
            name: abs(self._shortest_delta(now[name], target[name]))
            for name in self.arm_joint_names
        }

    def _robot_state(self, js: JointState) -> RobotState:
        state = RobotState()
        state.joint_state = copy.deepcopy(js)
        state.is_diff = False
        return state

    def _set_state(self, state: str, message: str) -> None:
        with self._state_lock:
            self._state = str(state)
            self._message = str(message)
        self._publish_status()

    def _load_home_configuration(
        self, *, raise_if_missing: bool = True
    ) -> Tuple[bool, str, Dict[str, float], Optional[float]]:
        path = Path(str(self.p["home_pose_path"])).expanduser().resolve()
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                names = [str(x) for x in data.get("joint_names", [])]
                positions = [float(x) for x in data.get("positions", [])]
                if len(names) == len(positions) and all(n in names for n in self.arm_joint_names):
                    mapping = {n: p for n, p in zip(names, positions)}
                    target = {n: float(mapping[n]) for n in self.arm_joint_names}
                    return True, "capture_file", target, float(data.get("captured_at_unix_s", 0.0))
            except Exception as exc:
                if raise_if_missing:
                    raise RuntimeError(f"Invalid home pose file {path}: {exc}") from exc
                self.get_logger().error(f"Invalid home pose file {path}: {exc}")

        configured = list(self.p["home_joint_positions"])
        if len(configured) == len(self.arm_joint_names):
            target = {n: float(v) for n, v in zip(self.arm_joint_names, configured)}
            return True, "yaml", target, None

        if raise_if_missing:
            raise RuntimeError(
                "HOME pose is not configured. Move the robot to the desired safe initial posture and call "
                f"'{self.p['capture_service']}', or provide 7 home_joint_positions in home.yaml."
            )
        return False, "", {}, None

    def _write_capture(self, js: JointState) -> Dict[str, float]:
        current = self._joint_map(js)
        positions = [float(current[name]) for name in self.arm_joint_names]
        payload = {
            "schema_version": 1,
            "planning_group": str(self.p["planning_group"]),
            "joint_names": list(self.arm_joint_names),
            "positions": positions,
            "captured_at_unix_s": time.time(),
            "note": "Safe task HOME/initial joint configuration captured from /joint_states.",
        }
        atomic_write_json(str(self.p["home_pose_path"]), payload)
        return {n: p for n, p in zip(self.arm_joint_names, positions)}

    def _wait_ready(self) -> None:
        timeout = 5.0
        if not self.plan_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"MoveIt service unavailable: {self.p['motion_plan_service']}")
        if not self.scene_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"MoveIt service unavailable: {self.p['get_planning_scene_service']}")
        if bool(self.p["ensure_table_collision_object"]):
            if not self.apply_scene_client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"MoveIt service unavailable: {self.p['apply_planning_scene_service']}")
        if bool(self.p["require_state_validity_service"]):
            if not self.validity_client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"MoveIt service unavailable: {self.p['state_validity_service']}")
        if not self.execute_action.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"MoveIt action unavailable: {self.p['execute_trajectory_action']}")

    def _table_exists(self) -> bool:
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        resp = self._wait_future(self.scene_client.call_async(req), 3.0)
        object_id = str(self.p["table_object_id"])
        return any(obj.id == object_id for obj in resp.scene.world.collision_objects)

    def _apply_table_collision_object(self) -> None:
        size_x = float(self.p["table_size_x_m"])
        size_y = float(self.p["table_size_y_m"])
        size_z = float(self.p["table_size_z_m"])
        if min(size_x, size_y, size_z) <= 0.0:
            raise RuntimeError("HOME table collision dimensions must all be positive")

        physical_top = float(self.p["table_physical_top_z_m"])
        surface_offset = float(self.p["table_collision_surface_offset_m"])
        collision_top = physical_top - surface_offset
        center_z = collision_top - 0.5 * size_z

        obj = CollisionObject()
        obj.header.frame_id = str(self.p["table_frame"])
        obj.id = str(self.p["table_object_id"])

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [size_x, size_y, size_z]

        pose = Pose()
        pose.orientation.w = 1.0
        pose.position.x = float(self.p["table_center_x_m"])
        pose.position.y = float(self.p["table_center_y_m"])
        pose.position.z = center_z

        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]

        req = ApplyPlanningScene.Request()
        req.scene = scene
        resp = self._wait_future(self.apply_scene_client.call_async(req), 3.0)
        if not bool(resp.success):
            raise RuntimeError("MoveIt rejected HOME table PlanningScene update")

        self.get_logger().info(
            "HOME safety table applied: "
            f"id={obj.id} frame={obj.header.frame_id} "
            f"size=({size_x:.3f},{size_y:.3f},{size_z:.3f})m "
            f"physical_top={physical_top:.4f}m collision_top={collision_top:.4f}m"
        )

    def _ensure_table_exists(self) -> None:
        if not bool(self.p["require_table_collision_object"]):
            return
        if self._table_exists():
            return
        if not bool(self.p["ensure_table_collision_object"]):
            raise RuntimeError(
                f"PlanningScene does not contain required table CollisionObject "
                f"'{self.p['table_object_id']}'"
            )

        self.get_logger().warn(
            f"PlanningScene table '{self.p['table_object_id']}' is missing; "
            "HomeExecutor will add its own configured safety table. "
            "No grasp planner call is required."
        )
        self._apply_table_collision_object()
        if not self._table_exists():
            raise RuntimeError(
                f"HOME table fallback was applied but CollisionObject "
                f"'{self.p['table_object_id']}' is still absent"
            )

    def _assert_current_state_valid(self, js: JointState) -> None:
        if not bool(self.p["require_state_validity_service"]):
            return
        req = GetStateValidity.Request()
        req.robot_state = self._robot_state(js)
        req.group_name = str(self.p["planning_group"])
        resp = self._wait_future(self.validity_client.call_async(req), 3.0)
        if not bool(resp.valid):
            contacts = [f"{c.contact_body_1}<->{c.contact_body_2}" for c in resp.contacts[:8]]
            raise RuntimeError(
                "Current robot state is invalid/colliding according to MoveIt"
                + (f": {contacts}" if contacts else "")
            )

    def _goal_constraints(self, target: Dict[str, float]) -> Constraints:
        c = Constraints()
        c.name = "open_vocab_home_joint_goal"
        tol = float(self.p["goal_joint_tolerance_rad"])
        for name in self.arm_joint_names:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(target[name])
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def _plan_home(self, current: JointState, target: Dict[str, float]):
        req = GetMotionPlan.Request()
        m = req.motion_plan_request
        m.group_name = str(self.p["planning_group"])
        m.start_state = self._robot_state(current)
        m.goal_constraints = [self._goal_constraints(target)]
        m.num_planning_attempts = int(self.p["planning_attempts"])
        m.allowed_planning_time = float(self.p["allowed_planning_time_s"])
        m.max_velocity_scaling_factor = float(self.p["velocity_scaling"])
        m.max_acceleration_scaling_factor = float(self.p["acceleration_scaling"])
        m.planner_id = str(self.p["planner_id"])
        resp = self._wait_future(
            self.plan_client.call_async(req),
            float(self.p["allowed_planning_time_s"]) + 2.0,
        )
        result = resp.motion_plan_response
        if result.error_code.val != MOVEIT_SUCCESS:
            raise RuntimeError(f"HOME OMPL plan failed, MoveIt code={result.error_code.val}")
        if not result.trajectory.joint_trajectory.points:
            raise RuntimeError("HOME OMPL returned an empty trajectory")
        return result.trajectory

    def _execute(self, trajectory) -> None:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = copy.deepcopy(trajectory)
        handle = self._wait_future(self.execute_action.send_goal_async(goal), 3.0)
        if handle is None or not handle.accepted:
            raise RuntimeError("MoveIt rejected HOME ExecuteTrajectory goal")
        with self._active_goal_lock:
            self._active_goal_handle = handle
        try:
            wrap = self._wait_future(
                handle.get_result_async(), float(self.p["execution_timeout_s"])
            )
        finally:
            with self._active_goal_lock:
                self._active_goal_handle = None
        if wrap is None:
            raise RuntimeError("HOME ExecuteTrajectory returned no result")
        if wrap.result.error_code.val != MOVEIT_SUCCESS:
            raise RuntimeError(
                f"HOME ExecuteTrajectory failed: action_status={wrap.status}, "
                f"MoveIt code={wrap.result.error_code.val}"
            )

    def _stop(self) -> None:
        event = String()
        event.data = "stop"
        self.trajectory_event_pub.publish(event)
        with self._active_goal_lock:
            handle = self._active_goal_handle
        if handle is not None:
            try:
                self._wait_future(handle.cancel_goal_async(), 2.0)
            except Exception as exc:
                self.get_logger().error(f"Unable to cancel active HOME trajectory: {exc}")

    def _status_payload(self) -> Dict[str, object]:
        configured, source, target, captured_at = self._load_home_configuration(raise_if_missing=False)
        current_error = None
        at_home = False
        try:
            js = self._current_joint_state()
            if configured:
                errors = self._joint_errors(js, target)
                current_error = max(errors.values()) if errors else None
                at_home = current_error is not None and current_error <= float(self.p["already_home_tolerance_rad"])
        except Exception:
            pass
        with self._state_lock:
            state = self._state
            message = self._message
        return {
            "state": state,
            "message": message,
            "configured": configured,
            "home_source": source or None,
            "home_pose_path": str(self.p["home_pose_path"]),
            "captured_at_unix_s": captured_at,
            "execution_enabled": bool(self.p["enable_home_execution"]),
            "ensure_table_collision_object": bool(self.p["ensure_table_collision_object"]),
            "require_table_collision_object": bool(self.p["require_table_collision_object"]),
            "at_home": bool(at_home),
            "max_joint_error_rad": current_error,
            "joint_names": list(self.arm_joint_names),
        }

    def _publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(self._status_payload(), ensure_ascii=False)
        self.status_pub.publish(msg)

    def _on_capture(self, request, response):
        del request
        if not self._motion_lock.acquire(blocking=False):
            response.success = False
            response.message = "HOME executor busy; cannot capture while a HOME motion is active"
            return response
        try:
            js = self._current_joint_state()
            target = self._write_capture(js)
            self._set_state("CAPTURED", "HOME pose captured from current joint state")
            response.success = True
            response.message = (
                "HOME_CAPTURED: "
                + ", ".join(f"{n}={target[n]:.6f}" for n in self.arm_joint_names)
                + f"; saved={self.p['home_pose_path']}"
            )
        except Exception as exc:
            self._set_state("ERROR", f"HOME capture failed: {exc}")
            self.get_logger().error(f"HOME capture failed: {exc}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
        finally:
            self._motion_lock.release()
        return response

    def _on_go(self, request, response):
        del request
        if not self._motion_lock.acquire(blocking=False):
            response.success = False
            response.message = "HOME executor busy"
            return response
        try:
            if not bool(self.p["enable_home_execution"]):
                raise RuntimeError("HOME execution locked: enable_home_execution=false")
            configured, source, target, _ = self._load_home_configuration(raise_if_missing=True)
            if not configured:
                raise RuntimeError("HOME pose not configured")

            self._set_state("CHECKING", f"Validating HOME target ({source})")
            self._wait_ready()
            current = self._current_joint_state()
            self._ensure_table_exists()
            self._assert_current_state_valid(current)

            errors = self._joint_errors(current, target)
            max_error = max(errors.values()) if errors else float("inf")
            if max_error <= float(self.p["already_home_tolerance_rad"]):
                self._set_state("AT_HOME", f"Already at HOME; max joint error={max_error:.4f}rad")
                response.success = True
                response.message = f"ALREADY_HOME: max_joint_error={max_error:.4f}rad"
                return response

            self._set_state("PLANNING", f"Planning current -> HOME; max initial joint error={max_error:.3f}rad")
            trajectory = self._plan_home(current, target)
            self._set_state("MOVING", "Executing collision-aware HOME trajectory")
            self._execute(trajectory)

            # Verify against live joint state, not merely the trajectory endpoint.
            time.sleep(0.10)
            final_js = self._current_joint_state()
            final_errors = self._joint_errors(final_js, target)
            final_max = max(final_errors.values()) if final_errors else float("inf")
            if final_max > float(self.p["final_home_tolerance_rad"]):
                raise RuntimeError(
                    f"HOME final verification failed: max joint error={final_max:.4f}rad > "
                    f"{float(self.p['final_home_tolerance_rad']):.4f}rad"
                )

            self._set_state("AT_HOME", f"HOME reached; max joint error={final_max:.4f}rad")
            response.success = True
            response.message = f"HOME_REACHED: max_joint_error={final_max:.4f}rad"
        except Exception as exc:
            self._set_state("ERROR", f"HOME motion failed: {exc}")
            self.get_logger().error(f"HOME motion failed: {exc}\n{traceback.format_exc()}")
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
        finally:
            self._motion_lock.release()
        return response

    def _on_abort(self, request, response):
        del request
        try:
            self._stop()
            self._set_state("ABORTED", "HOME software abort requested")
            response.success = True
            response.message = "HOME_ABORT_REQUESTED"
        except Exception as exc:
            response.success = False
            response.message = f"{type(exc).__name__}: {exc}"
        return response

    def _on_status(self, request, response):
        del request
        response.success = True
        response.message = json.dumps(self._status_payload(), ensure_ascii=False)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HomeExecutorNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop()
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
