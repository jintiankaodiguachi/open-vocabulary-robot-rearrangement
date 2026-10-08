import uuid
from .types import SceneObject, ObjectRole

class GlobalPerceptionPipeline:
    def __init__(self,ai_client,selector,projector,scene_store,default_role=ObjectRole.GRASPABLE):
        self.ai_client=ai_client; self.selector=selector; self.projector=projector; self.scene_store=scene_store; self.default_role=default_role

    def run(self,query,save_debug_image=True):
        r=self.ai_client.detect_segment(query,False,save_debug_image)
        d=self.selector.select(r.detections)
        x,y=self.projector.project(d.primary_uv[0],d.primary_uv[1],r.image_space_id)
        dbg=(self.ai_client.base_url+r.debug_image_url) if r.debug_image_url else None
        obj=SceneObject(str(uuid.uuid4()),query,d.label,self.default_role,d.confidence,d.sam_score,r.frame_id,r.timestamp,r.image_space_id,
                        d.bbox_xyxy,d.centroid_uv,d.support_uv,d.primary_uv,d.primary_point_type,(x,y),self.projector.output_frame,
                        False,None,d.mask_area_px,r.request_id,dbg)
        self.scene_store.upsert(obj,r.latency_ms)
        return obj
