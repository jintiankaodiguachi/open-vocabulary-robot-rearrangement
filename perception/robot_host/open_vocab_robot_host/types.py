from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class ObjectRole(str, Enum):
    GRASPABLE="GRASPABLE"; RECEPTACLE="RECEPTACLE"; SUPPORT_SURFACE="SUPPORT_SURFACE"; OBSTACLE="OBSTACLE"; UNKNOWN="UNKNOWN"

def _pt(v):
    if v is None: return None
    if not isinstance(v,(list,tuple)) or len(v)!=2: raise ValueError(f"invalid point: {v!r}")
    return (float(v[0]),float(v[1]))

@dataclass(frozen=True)
class Detection:
    detection_id:int; label:str; confidence:float; sam_score:Optional[float]
    bbox_xyxy:Tuple[float,float,float,float]; centroid_uv:Optional[Tuple[float,float]]
    support_uv:Optional[Tuple[float,float]]; primary_uv:Tuple[float,float]
    primary_point_type:str; mask_area_px:int

    @classmethod
    def from_dict(cls,d:Dict[str,Any]):
        b=d.get("bbox_xyxy")
        if not isinstance(b,(list,tuple)) or len(b)!=4: raise ValueError("invalid bbox_xyxy")
        p=_pt(d.get("primary_uv")) or _pt(d.get("centroid_uv"))
        if p is None: raise ValueError("missing primary_uv/centroid_uv")
        return cls(int(d.get("id",-1)),str(d.get("label","")),float(d.get("confidence",0)),
                   None if d.get("sam_score") is None else float(d["sam_score"]),
                   tuple(float(x) for x in b),_pt(d.get("centroid_uv")),_pt(d.get("support_uv")),
                   p,str(d.get("primary_point_type","mask_centroid")),int(d.get("mask_area_px",0)))

@dataclass(frozen=True)
class AIFrameResult:
    request_id:str; query:str; frame_id:int; timestamp:float; width:int; height:int
    image_space_id:str; detections:List[Detection]; latency_ms:Dict[str,float]; debug_image_url:Optional[str]

@dataclass
class SceneObject:
    object_id:str; query:str; label:str; role:ObjectRole; confidence:float; sam_score:Optional[float]
    frame_id:int; timestamp:float; image_space_id:str
    bbox_xyxy:Tuple[float,float,float,float]; centroid_uv:Optional[Tuple[float,float]]
    support_uv:Optional[Tuple[float,float]]; primary_uv:Tuple[float,float]; primary_point_type:str
    coarse_base_xy:Tuple[float,float]; coarse_base_frame:str
    has_precise_3d_pose:bool=False; precise_position_xyz:Optional[Tuple[float,float,float]]=None
    mask_area_px:int=0; source_request_id:str=""; debug_image_url:Optional[str]=None
    def to_dict(self):
        d=asdict(self); d["role"]=self.role.value; return d
