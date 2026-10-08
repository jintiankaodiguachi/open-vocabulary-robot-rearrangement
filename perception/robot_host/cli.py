#!/usr/bin/env python3
import argparse,json,logging,sys
from pathlib import Path
import yaml
from open_vocab_robot_host.ai_client import AIClient
from open_vocab_robot_host.errors import RobotHostError
from open_vocab_robot_host.homography import HomographyProjector
from open_vocab_robot_host.pipeline import GlobalPerceptionPipeline
from open_vocab_robot_host.scene_store import SceneStore
from open_vocab_robot_host.selector import DetectionSelector,SelectionPolicy
from open_vocab_robot_host.types import ObjectRole

def load(path):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def build(cfg):
    a=cfg["ai_service"]; s=cfg.get("selection",{}); sc=cfg.get("scene",{})
    client=AIClient(a["base_url"],a.get("expected_image_space_id","UNDISTORTED_1280x720"),
                    a.get("expected_width",1280),a.get("expected_height",720),a.get("connect_timeout_s",2),
                    a.get("read_timeout_s",20),a.get("retry_count",1),a.get("retry_backoff_s",0.25))
    selector=DetectionSelector(SelectionPolicy(float(s.get("min_confidence",0.50)),int(s.get("min_mask_area_px",300)),
                                                bool(s.get("reject_ambiguous",True)),float(s.get("ambiguity_margin",0.10))))
    projector=HomographyProjector.from_config(cfg["homography"])
    store=SceneStore(sc.get("output_dir","./outputs"))
    return GlobalPerceptionPipeline(client,selector,projector,store,ObjectRole(str(sc.get("default_role","GRASPABLE")).upper()))

def main():
    p=argparse.ArgumentParser(); p.add_argument("query",nargs="?"); p.add_argument("--config",default="config.yaml"); p.add_argument("--health",action="store_true")
    p.add_argument("--json",action="store_true"); p.add_argument("--no-debug-image",action="store_true"); args=p.parse_args()
    try:
        pipe=build(load(args.config))
        if args.health: print(json.dumps(pipe.ai_client.health(),ensure_ascii=False,indent=2)); return 0
        if not args.query: p.error("query required")
        o=pipe.run(args.query,not args.no_debug_image)
        if args.json: print(json.dumps(o.to_dict(),ensure_ascii=False,indent=2))
        else:
            print("\n"+"="*66)
            print("Stage 2A — Global Open-Vocabulary Perception")
            print("="*66)
            print(f"query               : {o.query}")
            print(f"label               : {o.label}")
            print(f"confidence          : {o.confidence:.4f}")
            print(f"frame_id            : {o.frame_id}")
            print(f"image_space_id      : {o.image_space_id}")
            print(f"primary_point_type  : {o.primary_point_type}")
            print(f"primary_uv          : ({o.primary_uv[0]:.2f}, {o.primary_uv[1]:.2f}) px")
            print(f"coarse_base_xy      : ({o.coarse_base_xy[0]:.4f}, {o.coarse_base_xy[1]:.4f}) m")
            print(f"base_frame          : {o.coarse_base_frame}")
            print(f"precise_3d          : {o.has_precise_3d_pose}")
            if o.debug_image_url: print(f"debug_image         : {o.debug_image_url}")
            print("="*66+"\n")
        return 0
    except RobotHostError as e:
        logging.error("%s",e); return 2
    except Exception:
        logging.exception("fatal"); return 3

if __name__=="__main__": sys.exit(main())
