from dataclasses import dataclass
from .errors import DetectionError, AmbiguousTargetError

@dataclass(frozen=True)
class SelectionPolicy:
    min_confidence:float=0.50
    min_mask_area_px:int=300
    reject_ambiguous:bool=True
    ambiguity_margin:float=0.10

class DetectionSelector:
    def __init__(self,policy): self.policy=policy
    def select(self,detections):
        c=[d for d in detections if d.confidence>=self.policy.min_confidence and d.mask_area_px>=self.policy.min_mask_area_px]
        if not c: raise DetectionError("no valid detection after confidence/mask filtering")
        c.sort(key=lambda d:(d.confidence,d.sam_score if d.sam_score is not None else -1,d.mask_area_px),reverse=True)
        if self.policy.reject_ambiguous and len(c)>1:
            margin=c[0].confidence-c[1].confidence
            if margin<self.policy.ambiguity_margin:
                raise AmbiguousTargetError(f"ambiguous target: confidence margin {margin:.3f} < {self.policy.ambiguity_margin:.3f}")
        return c[0]
