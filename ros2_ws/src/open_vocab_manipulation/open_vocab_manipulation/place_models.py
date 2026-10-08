from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from .models import PoseSpec


class PlaceState(str, Enum):
    IDLE = "IDLE"
    DESTINATION_LOCALIZED = "DESTINATION_LOCALIZED"
    PREPARED = "PREPARED"
    MOVING_PREPLACE = "MOVING_PREPLACE"
    AT_PREPLACE = "AT_PREPLACE"
    DESCENDING = "DESCENDING"
    AT_RELEASE = "AT_RELEASE"
    RELEASING = "RELEASING"
    RETREATING = "RETREATING"
    PLACED = "PLACED"
    VERIFIED = "VERIFIED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PlacePlan:
    source_label: str
    source_object_id: str
    destination_query: str
    destination_label: str
    destination_xy_m: List[float]
    desired_grasp_center_xyz_m: List[float]
    grasp_center_offset_base_m: List[float]
    preplace_pose: PoseSpec
    release_pose: PoseSpec
    table_top_z_m: float
    destination_rim_height_m: float
    release_grasp_center_above_rim_m: float
    preplace_clearance_m: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_label": self.source_label,
            "source_object_id": self.source_object_id,
            "destination_query": self.destination_query,
            "destination_label": self.destination_label,
            "destination_xy_m": list(self.destination_xy_m),
            "desired_grasp_center_xyz_m": list(self.desired_grasp_center_xyz_m),
            "grasp_center_offset_base_m": list(self.grasp_center_offset_base_m),
            "preplace_goal_link_pose": {
                "frame_id": self.preplace_pose.frame_id,
                "position": list(self.preplace_pose.position),
                "quaternion_xyzw": list(self.preplace_pose.quaternion_xyzw),
            },
            "release_goal_link_pose": {
                "frame_id": self.release_pose.frame_id,
                "position": list(self.release_pose.position),
                "quaternion_xyzw": list(self.release_pose.quaternion_xyzw),
            },
            "table_top_z_m": self.table_top_z_m,
            "destination_rim_height_m": self.destination_rim_height_m,
            "release_grasp_center_above_rim_m": self.release_grasp_center_above_rim_m,
            "preplace_clearance_m": self.preplace_clearance_m,
        }
