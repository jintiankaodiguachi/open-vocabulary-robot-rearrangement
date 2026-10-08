#!/usr/bin/env python3
import time, cv2, numpy as np, requests
class RK3588CameraClient:
    def __init__(self,url,space,width=1280,height=720,timeout=3.0):
        self.url=url; self.space=space; self.width=width; self.height=height; self.timeout=timeout
    def fetch(self):
        t0=time.perf_counter(); r=requests.get(self.url,timeout=self.timeout); r.raise_for_status()
        network_ms=(time.perf_counter()-t0)*1000
        got_space=r.headers.get("X-Image-Space","")
        if got_space and got_space!=self.space:
            raise RuntimeError(f"image_space mismatch: expected {self.space}, got {got_space}")
        img=cv2.imdecode(np.frombuffer(r.content,np.uint8),cv2.IMREAD_COLOR)
        if img is None: raise RuntimeError("JPEG decode failed")
        h,w=img.shape[:2]
        if (w,h)!=(self.width,self.height):
            raise RuntimeError(f"resolution mismatch: expected {self.width}x{self.height}, got {w}x{h}")
        meta={"frame_id":int(r.headers.get("X-Frame-Id",-1)),
              "timestamp":float(r.headers.get("X-Timestamp",0)),
              "width":w,"height":h,"encoding":r.headers.get("X-Encoding","jpeg"),
              "image_space_id":got_space or self.space}
        return img,meta,network_ms
