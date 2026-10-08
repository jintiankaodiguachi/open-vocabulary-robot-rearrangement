import csv, json, threading
from pathlib import Path

class SceneStore:
    def __init__(self,output_dir):
        self.dir=Path(output_dir).expanduser().resolve(); self.dir.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock(); self.latest={}
        self.latest_path=self.dir/"latest_scene.json"; self.history=self.dir/"scene_history.jsonl"; self.metrics=self.dir/"metrics.csv"

    def upsert(self,obj,latency):
        with self.lock:
            self.latest[obj.label.lower()]=obj
            tmp=self.latest_path.with_suffix(".tmp")
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump({"objects":[o.to_dict() for o in self.latest.values()]},f,ensure_ascii=False,indent=2)
            tmp.replace(self.latest_path)
            with open(self.history,"a",encoding="utf-8") as f:
                f.write(json.dumps(obj.to_dict(),ensure_ascii=False,separators=(",",":"))+"\n")
            fields=["frame_id","timestamp","query","label","confidence","sam_score","primary_u","primary_v","coarse_x","coarse_y","mask_area_px",
                    "network_fetch_ms","grounding_ms","sam_ms","postprocess_ms","total_model_ms","total_ms"]
            row={"frame_id":obj.frame_id,"timestamp":obj.timestamp,"query":obj.query,"label":obj.label,"confidence":obj.confidence,"sam_score":obj.sam_score,
                 "primary_u":obj.primary_uv[0],"primary_v":obj.primary_uv[1],"coarse_x":obj.coarse_base_xy[0],"coarse_y":obj.coarse_base_xy[1],
                 "mask_area_px":obj.mask_area_px,**{k:latency.get(k) for k in fields if k.endswith("_ms")}}
            exists=self.metrics.exists()
            with open(self.metrics,"a",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=fields)
                if not exists: w.writeheader()
                w.writerow(row)
