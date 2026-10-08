#!/usr/bin/env python3
import argparse,json,time,uuid
from pathlib import Path
import cv2,torch,yaml
from flask import Flask,jsonify,request,send_from_directory
from camera_client import RK3588CameraClient
from grounded_sam2_engine import EngineConfig,GroundedSAM2Engine
from wrist_upload_api import register_wrist_upload_api

ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); args=ap.parse_args()
cfg=yaml.safe_load(open(args.config,"r",encoding="utf-8"))
m=cfg["model"]; c=cfg["camera"]; s=cfg["service"]; o=cfg.get("output",{})
root=Path(m["repo_root"]).resolve()
out=Path(o.get("dir",root/"outputs/open_vocab_service")).resolve(); out.mkdir(parents=True,exist_ok=True)

engine=GroundedSAM2Engine(EngineConfig(
    repo_root=str(root),sam2_checkpoint=m["sam2_checkpoint"],sam2_model_config=m["sam2_model_config"],
    grounding_dino_config=m["grounding_dino_config"],grounding_dino_checkpoint=m["grounding_dino_checkpoint"],
    device=m.get("device","cuda"),box_threshold=float(m.get("box_threshold",0.35)),
    text_threshold=float(m.get("text_threshold",0.25)),multimask_output=bool(m.get("multimask_output",False)),
    support_band_ratio=float(m.get("support_band_ratio",0.03)),enable_tf32=bool(m.get("enable_tf32",True))))
camera=RK3588CameraClient(c["latest_jpeg_url"],c.get("expected_image_space_id","UNDISTORTED_1280x720"),
                          int(c.get("expected_width",1280)),int(c.get("expected_height",720)),float(c.get("timeout_s",3.0)))
app=Flask(__name__)

register_wrist_upload_api(
    app,
    engine,
    out
)

@app.get("/health")
def health():
    return jsonify({"status":"ok","service":"open_vocab_grounded_sam2","torch":torch.__version__,
                    "cuda_available":torch.cuda.is_available(),
                    "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "expected_image_space_id":camera.space})

@app.post("/api/v1/detect_segment")
def detect_segment():
    body=request.get_json(silent=True) or {}; prompt=str(body.get("text_prompt","")).strip()
    if not prompt: return jsonify({"error":"text_prompt is required"}),400
    rid=str(uuid.uuid4())
    try:
        t0=time.perf_counter()
        img,meta,net=camera.fetch()
        result,dbg=engine.infer(img,prompt,bool(body.get("include_mask_rle",True)),
                                bool(body.get("save_debug_image",True)))
        debug_url=None
        if dbg is not None:
            name=f"{rid}_debug.jpg"; cv2.imwrite(str(out/name),dbg); cv2.imwrite(str(out/"latest_detection.jpg"),dbg)
            debug_url=f"/outputs/{name}"
        result.update({"request_id":rid,**meta,"debug_image_url":debug_url})
        result["latency_ms"]["network_fetch_ms"]=net
        result["latency_ms"]["total_ms"]=(time.perf_counter()-t0)*1000
        json.dump(result,open(out/"latest_result.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("detect_segment failed")
        return jsonify({"error":type(e).__name__,"message":str(e),"request_id":rid}),500

@app.get("/outputs/<path:name>")
def outputs(name): return send_from_directory(str(out),name,as_attachment=False)

if __name__=="__main__":
    app.run(host=s.get("host","0.0.0.0"),port=int(s.get("port",8090)),threaded=True,debug=False,use_reloader=False)
