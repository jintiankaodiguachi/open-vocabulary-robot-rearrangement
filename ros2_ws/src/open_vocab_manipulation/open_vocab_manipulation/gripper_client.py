from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from control_msgs.action import GripperCommand
from rclpy.action import ActionClient


@dataclass(frozen=True)
class GripperResult:
    position: float
    effort: float
    stalled: bool
    reached_goal: bool
    action_status: int


class RobotiqGripperClient:
    """ROS 2 Humble Robotiq/Kortex GripperCommand action wrapper.

    Kinova ros2_kortex exposes the 2F-85 controller at
    /robotiq_gripper_controller/gripper_cmd. In that driver, controller
    position is approximately 0.0=open and 0.8=closed; it is a controller joint
    coordinate, NOT the GraspNet physical jaw width in metres.
    """

    def __init__(self, node, callback_group, action_name: str, timeout_s: float):
        self.node = node
        self.timeout_s = float(timeout_s)
        self.client = ActionClient(
            node,
            GripperCommand,
            action_name,
            callback_group=callback_group,
        )
        self._active_goal_handle = None
        self._lock = threading.Lock()

    @staticmethod
    def _wait_future(future, timeout_s: float):
        deadline = time.monotonic() + float(timeout_s)
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("Gripper action timed out")
            time.sleep(0.005)
        if future.exception() is not None:
            raise future.exception()
        return future.result()

    def wait_until_ready(self, timeout_s: float = 5.0) -> None:
        if not self.client.wait_for_server(timeout_sec=timeout_s):
            raise RuntimeError("Robotiq gripper action server unavailable")

    def command(self, position: float, max_effort: float) -> GripperResult:
        if not self.client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError("Robotiq gripper action server unavailable")
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        goal_handle = self._wait_future(self.client.send_goal_async(goal), 3.0)
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Robotiq gripper rejected command")
        with self._lock:
            self._active_goal_handle = goal_handle
        try:
            wrapped = self._wait_future(goal_handle.get_result_async(), self.timeout_s)
        finally:
            with self._lock:
                self._active_goal_handle = None
        if wrapped is None or wrapped.result is None:
            raise RuntimeError("Robotiq gripper returned no result")
        r = wrapped.result
        return GripperResult(
            position=float(r.position),
            effort=float(r.effort),
            stalled=bool(r.stalled),
            reached_goal=bool(r.reached_goal),
            action_status=int(wrapped.status),
        )

    def cancel(self) -> None:
        with self._lock:
            handle = self._active_goal_handle
        if handle is not None:
            try:
                self._wait_future(handle.cancel_goal_async(), 2.0)
            except Exception as exc:
                self.node.get_logger().error(f"Failed to cancel gripper action: {exc}")
