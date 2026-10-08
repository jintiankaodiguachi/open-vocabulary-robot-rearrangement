from __future__ import annotations
from pathlib import Path
import numpy as np
from .errors import ConfigurationError, GeometryError

class HomographyProjector:
    def __init__(self,H,image_space_id,output_frame="base_link",output_scale=1.0,workspace=None):
        H=np.asarray(H,dtype=np.float64)
        if H.shape!=(3,3) or not np.isfinite(H).all(): raise ConfigurationError("Homography must be finite 3x3")
        if abs(np.linalg.det(H))<1e-12: raise ConfigurationError("Homography is singular")
        self.H=H; self.image_space_id=str(image_space_id); self.output_frame=str(output_frame)
        self.output_scale=float(output_scale); self.workspace=workspace or {}

    @classmethod
    def from_config(cls,cfg):
        if not bool(cfg.get("configured",False)): raise ConfigurationError("set homography.configured=true and provide the REAL calibrated H")
        src=str(cfg.get("source","inline")).lower()
        if src=="inline":
            H=cfg.get("matrix")
            if H is None: raise ConfigurationError("missing homography.matrix")
        elif src=="npy":
            p=Path(str(cfg.get("npy_path",""))).expanduser()
            if not p.exists(): raise ConfigurationError(f"missing H npy: {p}")
            H=np.load(str(p))
        else: raise ConfigurationError("homography.source must be inline or npy")
        return cls(H,cfg.get("image_space_id","UNDISTORTED_1280x720"),
                   cfg.get("output_frame","base_link"),cfg.get("output_scale",1.0),cfg.get("workspace",{}))

    def project(self,u,v,image_space_id):
        if image_space_id!=self.image_space_id: raise GeometryError("Homography/image-space mismatch")
        q=self.H@np.array([float(u),float(v),1.0],dtype=np.float64)
        if abs(float(q[2]))<1e-10: raise GeometryError("invalid homogeneous denominator")
        x=float(q[0]/q[2])*self.output_scale; y=float(q[1]/q[2])*self.output_scale
        if not np.isfinite([x,y]).all(): raise GeometryError("non-finite projected XY")
        w=self.workspace
        checks=(("x_min",x,lambda a,b:a<b),("x_max",x,lambda a,b:a>b),("y_min",y,lambda a,b:a<b),("y_max",y,lambda a,b:a>b))
        for key,val,bad in checks:
            if key in w and bad(val,float(w[key])): raise GeometryError(f"{key} violated: value={val:.4f}, limit={w[key]}")
        return x,y
