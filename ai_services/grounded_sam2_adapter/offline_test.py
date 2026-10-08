#!/usr/bin/env python3
import argparse,json,cv2,yaml
from pathlib import Path
from grounded_sam2_engine import EngineConfig,GroundedSAM2Engine
ap=argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("text_prompt"); ap.add_argument("--config",default="config.yaml"); a=ap.parse_args()
cfg=yaml.safe_load(open(a.config,"r",encoding="utf-8")); m=cfg["model"]; root=Path(m["repo_root"]).resolve()
e=GroundedSAM2Engine(EngineConfig(str(root),m["sam2_checkpoint"],m["sam2_model_config"],m["grounding_dino_config"],
    m["grounding_dino_checkpoint"],m.get("device","cuda"),float(m.get("box_threshold",0.35)),
    float(m.get("text_threshold",0.25)),bool(m.get("multimask_output",False)),float(m.get("support_band_ratio",0.03)),
    bool(m.get("enable_tf32",True))))
img=cv2.imread(a.image)
if img is None: raise RuntimeError(f"cannot read {a.image}")
r,dbg=e.infer(img,a.text_prompt,False,True); print(json.dumps(r,ensure_ascii=False,indent=2))
out=root/"outputs/offline_test"; out.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(out/"debug.jpg"),dbg)
print("debug:",out/"debug.jpg")
