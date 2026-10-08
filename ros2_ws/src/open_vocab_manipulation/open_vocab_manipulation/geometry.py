from __future__ import annotations

import copy
import math
from typing import Iterable, Tuple

from geometry_msgs.msg import PoseStamped, TransformStamped

from .models import PoseSpec


def pose_spec_to_msg(spec: PoseSpec, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = spec.frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = spec.position[0]
    msg.pose.position.y = spec.position[1]
    msg.pose.position.z = spec.position[2]
    msg.pose.orientation.x = spec.quaternion_xyzw[0]
    msg.pose.orientation.y = spec.quaternion_xyzw[1]
    msg.pose.orientation.z = spec.quaternion_xyzw[2]
    msg.pose.orientation.w = spec.quaternion_xyzw[3]
    return msg


def transform_to_pose_stamped(tf: TransformStamped) -> PoseStamped:
    msg = PoseStamped()
    msg.header = tf.header
    msg.pose.position.x = tf.transform.translation.x
    msg.pose.position.y = tf.transform.translation.y
    msg.pose.position.z = tf.transform.translation.z
    msg.pose.orientation = tf.transform.rotation
    return msg


def euclidean3(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(av, bv)))


def euclidean2(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    return math.hypot(float(av[0]) - float(bv[0]), float(av[1]) - float(bv[1]))


def quaternion_angle_rad(q1: Iterable[float], q2: Iterable[float]) -> float:
    a = [float(v) for v in q1]
    b = [float(v) for v in q2]
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na <= 1e-12 or nb <= 1e-12:
        return float("inf")
    dot = abs(sum(x * y for x, y in zip(a, b)) / (na * nb))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def pose_error(actual: PoseStamped, target: PoseStamped) -> Tuple[float, float]:
    p1 = [actual.pose.position.x, actual.pose.position.y, actual.pose.position.z]
    p2 = [target.pose.position.x, target.pose.position.y, target.pose.position.z]
    q1 = [
        actual.pose.orientation.x,
        actual.pose.orientation.y,
        actual.pose.orientation.z,
        actual.pose.orientation.w,
    ]
    q2 = [
        target.pose.orientation.x,
        target.pose.orientation.y,
        target.pose.orientation.z,
        target.pose.orientation.w,
    ]
    return euclidean3(p1, p2), quaternion_angle_rad(q1, q2)


def translated_pose(pose: PoseStamped, dz: float) -> PoseStamped:
    out = PoseStamped()
    out.header = pose.header
    out.pose = copy.deepcopy(pose.pose)
    out.pose.position.z += float(dz)
    return out


def quaternion_multiply_xyzw(q1: Iterable[float], q2: Iterable[float]):
    """Hamilton product q1*q2 for ROS xyzw quaternions."""
    x1, y1, z1, w1 = [float(v) for v in q1]
    x2, y2, z2, w2 = [float(v) for v in q2]
    q = [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]
    n = math.sqrt(sum(v*v for v in q))
    if n <= 1e-12:
        raise ValueError("Quaternion multiplication produced a zero quaternion")
    return [v / n for v in q]


def quaternion_from_yaw(yaw_rad: float):
    h = 0.5 * float(yaw_rad)
    return [0.0, 0.0, math.sin(h), math.cos(h)]


def rotate_vector_about_z(vector: Iterable[float], yaw_rad: float):
    x, y, z = [float(v) for v in vector]
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return [c*x - s*y, s*x + c*y, z]
