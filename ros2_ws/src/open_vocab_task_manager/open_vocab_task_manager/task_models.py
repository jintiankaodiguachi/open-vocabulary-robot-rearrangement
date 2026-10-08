from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class TaskState(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    RESETTING = "RESETTING"
    MOVE_HOME_START = "MOVE_HOME_START"
    GLOBAL_SOURCE = "GLOBAL_SOURCE"
    WAIT_OBSERVATION = "WAIT_OBSERVATION"
    LOCAL_SOURCE = "LOCAL_SOURCE"
    PLAN_PICK = "PLAN_PICK"
    PREPARE_PICK = "PREPARE_PICK"
    MOVE_PREGRASP = "MOVE_PREGRASP"
    PICK_DRY_RUN = "PICK_DRY_RUN"
    GRASP_AND_LIFT = "GRASP_AND_LIFT"
    CONFIGURE_DESTINATION = "CONFIGURE_DESTINATION"
    LOCATE_DESTINATION = "LOCATE_DESTINATION"
    PREPARE_PLACE = "PREPARE_PLACE"
    MOVE_PREPLACE = "MOVE_PREPLACE"
    RELEASE_AND_RETREAT = "RELEASE_AND_RETREAT"
    VERIFY_PLACE = "VERIFY_PLACE"
    RETURN_HOME = "RETURN_HOME"
    DONE = "DONE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ServiceResult:
    success: bool
    message: str


@dataclass(frozen=True)
class DestinationProfile:
    name: str
    query: str
    aliases: tuple[str, ...]
    inherit_place_baseline: bool
    require_explicit_geometry: bool
    values: Dict[str, Any]
    matched_override: bool = True
