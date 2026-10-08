from __future__ import annotations
import time, requests
from .errors import AIServiceError
from .types import AIFrameResult, Detection

class AIClient:
    def __init__(self,base_url,expected_image_space_id,expected_width=1280,expected_height=720,
                 connect_timeout_s=2.0,read_timeout_s=20.0,retry_count=1,retry_backoff_s=0.25):
        self.base_url=base_url.rstrip("/"); self.expected_image_space_id=expected_image_space_id
        self.expected_width=int(expected_width); self.expected_height=int(expected_height)
        self.connect_timeout_s=float(connect_timeout_s); self.read_timeout_s=float(read_timeout_s)
        self.retry_count=max(0,int(retry_count)); self.retry_backoff_s=float(retry_backoff_s)
        self.session=requests.Session()

    def health(self):
        try:
            r=self.session.get(self.base_url+"/health",timeout=(self.connect_timeout_s,self.read_timeout_s))
            r.raise_for_status(); return r.json()
        except Exception as e: raise AIServiceError(f"A6000 /health failed: {e}") from e

    def detect_segment(self,text_prompt,include_mask_rle=False,save_debug_image=True):
        payload={"text_prompt":str(text_prompt).strip(),"include_mask_rle":bool(include_mask_rle),"save_debug_image":bool(save_debug_image)}
        if not payload["text_prompt"]: raise ValueError("empty text_prompt")
        last=None
        for attempt in range(self.retry_count+1):
            try:
                r=self.session.post(self.base_url+"/api/v1/detect_segment",json=payload,
                                    timeout=(self.connect_timeout_s,self.read_timeout_s))
                if r.status_code in (503,504): raise AIServiceError(f"transient HTTP {r.status_code}: {r.text[:300]}")
                if r.status_code!=200: raise AIServiceError(f"HTTP {r.status_code}: {r.text[:1000]}")
                return self._parse(r.json())
            except (requests.ConnectionError,requests.Timeout,AIServiceError) as e:
                last=e
                if attempt>=self.retry_count: break
                time.sleep(self.retry_backoff_s*(2**attempt))
            except Exception as e:
                raise AIServiceError(f"A6000 response error: {e}") from e
        raise AIServiceError(f"detect_segment failed after {self.retry_count+1} attempt(s): {last}")

    def _parse(self,d):
        for k in ("request_id","query","frame_id","timestamp","width","height","image_space_id","detections"):
            if k not in d: raise ValueError(f"missing field {k}")
        w,h=int(d["width"]),int(d["height"]); space=str(d["image_space_id"])
        if (w,h)!=(self.expected_width,self.expected_height):
            raise ValueError(f"resolution mismatch: {w}x{h}")
        if space!=self.expected_image_space_id:
            raise ValueError(f"image_space mismatch: {space}")
        dets=[Detection.from_dict(x) for x in d["detections"]]
        lat={str(k):float(v) for k,v in d.get("latency_ms",{}).items() if isinstance(v,(int,float))}
        return AIFrameResult(str(d["request_id"]),str(d["query"]),int(d["frame_id"]),float(d["timestamp"]),
                             w,h,space,dets,lat,d.get("debug_image_url"))
