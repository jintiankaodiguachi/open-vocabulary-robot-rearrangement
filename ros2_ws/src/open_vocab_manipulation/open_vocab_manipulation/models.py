from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class PickState(str, Enum):
    IDLE = "IDLE"
    PREPARED = "PREPARED"
    MOVING_PREGRASP = "MOVING_PREGRASP"
    AT_PREGRASP = "AT_PREGRASP"
    REVALIDATING = "REVALIDATING"
    REVALIDATED = "REVALIDATED"
    APPROACHING_DRY = "APPROACHING_DRY"
    AT_GRASP_DRY = "AT_GRASP_DRY"
    RETREATING = "RETREATING"
    GRASPING = "GRASPING"
    LIFTING = "LIFTING"
    PICKED = "PICKED"
    INVALIDATED = "INVALIDATED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PoseSpec:
    frame_id: str
    position: List[float]
    quaternion_xyzw: List[float]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PoseSpec":
        frame_id = str(data.get("frame_id", ""))
        position = [float(v) for v in data.get("position", [])]
        quat = [float(v) for v in data.get("quaternion_xyzw", [])]
        if not frame_id:
            raise ValueError("pose frame_id is empty")
        if len(position) != 3:
            raise ValueError(f"pose position must contain 3 values, got {len(position)}")
        if len(quat) != 4:
            raise ValueError(f"pose quaternion_xyzw must contain 4 values, got {len(quat)}")
        return PoseSpec(frame_id=frame_id, position=position, quaternion_xyzw=quat)


@dataclass(frozen=True)
class SelectedGrasp:
    rank: int
    model_score: float
    selection_score: float
    width_m: float
    suggested_open_width_m: float
    cartesian_fraction: float
    max_joint_delta_rad: float
    joint_l1_delta_rad: float
    approach_axis_base: List[float]
    pregrasp_pose: PoseSpec
    grasp_pose: PoseSpec

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SelectedGrasp":
        if not bool(data.get("feasible", False)):
            raise ValueError("selected grasp is not marked feasible")
        candidate = data.get("candidate") or {}
        approach = [float(v) for v in data.get("approach_axis_base", [])]
        if len(approach) != 3:
            raise ValueError("selected approach_axis_base must contain 3 values")
        return SelectedGrasp(
            rank=int(data.get("rank", candidate.get("rank", -1))),
            model_score=float(candidate.get("model_score", 0.0)),
            selection_score=float(data.get("selection_score", 0.0)),
            width_m=float(candidate.get("width_m", 0.0)),
            suggested_open_width_m=float(data.get("suggested_open_width_m", 0.0)),
            cartesian_fraction=float(data.get("cartesian_fraction", 0.0)),
            max_joint_delta_rad=float(data.get("max_joint_delta_rad", 0.0)),
            joint_l1_delta_rad=float(data.get("joint_l1_delta_rad", 0.0)),
            approach_axis_base=approach,
            pregrasp_pose=PoseSpec.from_dict(data["pregrasp_goal_link_pose"]),
            grasp_pose=PoseSpec.from_dict(data["grasp_goal_link_pose"]),
        )


@dataclass(frozen=True)
class GraspPlan:
    label: str
    object_id: str
    base_frame: str
    planning_group: str
    goal_link: str
    created_at_unix_s: Optional[float]
    source_npz_mtime_ns: Optional[int]
    source_stamp_ns: Optional[int]
    source_centroid_base_xyz_m: Optional[List[float]]
    table_collision_enabled: bool
    table_object_id: str
    physical_table_top_z_m: Optional[float]
    selected: SelectedGrasp

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GraspPlan":
        if data.get("status") != "ok":
            raise ValueError(f"grasp plan status is not ok: {data.get('status')}")
        if data.get("selected") is None:
            raise ValueError("grasp plan has no selected candidate")
        tc = data.get("table_collision") or {}
        centroid = data.get("source_centroid_base_xyz_m")
        if centroid is not None:
            centroid = [float(v) for v in centroid]
            if len(centroid) != 3:
                raise ValueError("source_centroid_base_xyz_m must have 3 values")
        created = data.get("created_at_unix_s")
        npz_mtime = data.get("source_npz_mtime_ns")
        source_stamp = data.get("source_stamp_ns")
        return GraspPlan(
            label=str(data.get("label", "")).strip(),
            object_id=str(data.get("object_id", "")).strip(),
            base_frame=str(data.get("base_frame", "")).strip(),
            planning_group=str(data.get("planning_group", "")).strip(),
            goal_link=str(data.get("goal_link", "")).strip(),
            created_at_unix_s=float(created) if created is not None else None,
            source_npz_mtime_ns=int(npz_mtime) if npz_mtime is not None else None,
            source_stamp_ns=int(source_stamp) if source_stamp is not None else None,
            source_centroid_base_xyz_m=centroid,
            table_collision_enabled=bool(tc.get("enabled", False)),
            table_object_id=str(tc.get("object_id", "")),
            physical_table_top_z_m=(
                float(tc["physical_top_z_m"])
                if tc.get("physical_top_z_m") is not None
                else None
            ),
            selected=SelectedGrasp.from_dict(data["selected"]),
        )


@dataclass(frozen=True)
class LocalObservation:
    object_id: str
    label: str
    centroid_base_xyz_m: List[float]
    table_z_m: Optional[float]
    valid_depth_ratio: Optional[float]
    cloud_points: Optional[int]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LocalObservation":
        if data.get("status") != "ok":
            raise ValueError(f"local perception status is not ok: {data.get('status')}")
        centroid = [float(v) for v in data.get("centroid_base_xyz_m", [])]
        if len(centroid) != 3:
            raise ValueError("local perception centroid_base_xyz_m must contain 3 values")
        return LocalObservation(
            object_id=str(data.get("object_id", "")).strip(),
            label=str(data.get("label", "")).strip(),
            centroid_base_xyz_m=centroid,
            table_z_m=float(data["table_z_m"]) if data.get("table_z_m") is not None else None,
            valid_depth_ratio=(
                float(data["valid_depth_ratio"])
                if data.get("valid_depth_ratio") is not None
                else None
            ),
            cloud_points=int(data["cloud_points"]) if data.get("cloud_points") is not None else None,
        )
