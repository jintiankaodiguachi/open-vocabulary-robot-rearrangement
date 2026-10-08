from __future__ import annotations

from typing import Sequence

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .models import PoseSpec


def _color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _point(xyz: Sequence[float]) -> Point:
    p = Point()
    p.x, p.y, p.z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return p


def make_place_markers(
    frame_id: str,
    stamp,
    destination_xy,
    desired_grasp_center_xyz,
    preplace: PoseSpec,
    release: PoseSpec,
    label: str,
) -> MarkerArray:
    arr = MarkerArray()

    delete = Marker()
    delete.action = Marker.DELETEALL
    arr.markers.append(delete)

    dest = Marker()
    dest.header.frame_id = frame_id
    dest.header.stamp = stamp
    dest.ns = "open_vocab_place"
    dest.id = 1
    dest.type = Marker.CYLINDER
    dest.action = Marker.ADD
    dest.pose.position.x = float(destination_xy[0])
    dest.pose.position.y = float(destination_xy[1])
    dest.pose.position.z = float(desired_grasp_center_xyz[2])
    dest.pose.orientation.w = 1.0
    dest.scale.x = 0.05
    dest.scale.y = 0.05
    dest.scale.z = 0.006
    dest.color = _color(0.1, 0.9, 0.2, 0.85)
    arr.markers.append(dest)

    anchor = Marker()
    anchor.header = dest.header
    anchor.ns = "open_vocab_place"
    anchor.id = 2
    anchor.type = Marker.SPHERE
    anchor.action = Marker.ADD
    anchor.pose.position.x = float(desired_grasp_center_xyz[0])
    anchor.pose.position.y = float(desired_grasp_center_xyz[1])
    anchor.pose.position.z = float(desired_grasp_center_xyz[2])
    anchor.pose.orientation.w = 1.0
    anchor.scale.x = anchor.scale.y = anchor.scale.z = 0.025
    anchor.color = _color(1.0, 0.55, 0.05, 1.0)
    arr.markers.append(anchor)

    line = Marker()
    line.header = dest.header
    line.ns = "open_vocab_place"
    line.id = 3
    line.type = Marker.ARROW
    line.action = Marker.ADD
    line.points = [_point(preplace.position), _point(release.position)]
    line.scale.x = 0.012
    line.scale.y = 0.024
    line.scale.z = 0.024
    line.color = _color(0.1, 0.75, 1.0, 0.95)
    arr.markers.append(line)

    text = Marker()
    text.header = dest.header
    text.ns = "open_vocab_place"
    text.id = 4
    text.type = Marker.TEXT_VIEW_FACING
    text.action = Marker.ADD
    text.pose.position.x = float(desired_grasp_center_xyz[0])
    text.pose.position.y = float(desired_grasp_center_xyz[1])
    text.pose.position.z = float(preplace.position[2]) + 0.05
    text.pose.orientation.w = 1.0
    text.scale.z = 0.035
    text.color = _color(1.0, 1.0, 1.0, 1.0)
    text.text = f"PLACE -> {label}"
    arr.markers.append(text)

    return arr
