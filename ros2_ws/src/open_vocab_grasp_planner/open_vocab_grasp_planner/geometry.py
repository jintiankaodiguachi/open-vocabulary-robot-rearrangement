from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from geometry_msgs.msg import PoseStamped


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("zero-length vector")
    return v / n


def project_to_so3(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    if np.linalg.det(out) < 0:
        U[:, -1] *= -1.0
        out = U @ Vt
    return out


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    R = project_to_so3(R)
    q = np.empty(4, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / s
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / s
        q[0] = (R[0, 2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q *= -1.0
    return q


def pose_stamped(frame_id: str, p: np.ndarray, R: np.ndarray, stamp) -> PoseStamped:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    q = rotation_matrix_to_quaternion_xyzw(R)
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
    return msg


def build_goal_link_poses(candidate: dict, frame_id: str, stamp,
                          R_grasp_to_goal: np.ndarray,
                          goal_link_to_grasp_center_m: float,
                          pregrasp_distance_m: float) -> Tuple[PoseStamped, PoseStamped, np.ndarray]:
    """
    GraspNet translation is the grasp-center point and local +X is approach.
    The MoveIt-controlled goal-link origin stays behind the physical grasp center.
    """
    p_center = np.asarray(candidate["translation_base_grasp_m"], dtype=np.float64).reshape(3)
    R_bg = project_to_so3(np.asarray(candidate["rotation_base_grasp"], dtype=np.float64).reshape(3, 3))
    approach = normalize(R_bg[:, 0])
    R_goal = project_to_so3(R_bg @ np.asarray(R_grasp_to_goal, dtype=np.float64).reshape(3, 3))
    p_goal = p_center - approach * float(goal_link_to_grasp_center_m)
    p_pre = p_goal - approach * float(pregrasp_distance_m)
    return pose_stamped(frame_id, p_pre, R_goal, stamp), pose_stamped(frame_id, p_goal, R_goal, stamp), approach


def shortest_angular_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(b - a), math.cos(b - a))
