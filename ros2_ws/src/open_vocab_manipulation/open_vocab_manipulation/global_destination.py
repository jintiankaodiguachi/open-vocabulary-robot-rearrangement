from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DestinationObservation:
    query: str
    label: str
    confidence: float
    sam_score: Optional[float]
    frame_id: int
    timestamp: float
    image_space_id: str
    width: int
    height: int
    bbox_xyxy: Tuple[float, float, float, float]
    centroid_uv: Tuple[float, float]
    primary_uv: Tuple[float, float]
    primary_point_type: str
    mask_area_px: int
    base_xy_m: Tuple[float, float]
    request_id: str
    debug_image_url: Optional[str]
    created_at_unix_s: float

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["bbox_xyxy"] = list(self.bbox_xyxy)
        out["centroid_uv"] = list(self.centroid_uv)
        out["primary_uv"] = list(self.primary_uv)
        out["base_xy_m"] = list(self.base_xy_m)
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DestinationObservation":
        return DestinationObservation(
            query=str(data["query"]),
            label=str(data["label"]),
            confidence=float(data["confidence"]),
            sam_score=(None if data.get("sam_score") is None else float(data["sam_score"])),
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            image_space_id=str(data["image_space_id"]),
            width=int(data["width"]),
            height=int(data["height"]),
            bbox_xyxy=tuple(float(v) for v in data["bbox_xyxy"]),
            centroid_uv=tuple(float(v) for v in data["centroid_uv"]),
            primary_uv=tuple(float(v) for v in data["primary_uv"]),
            primary_point_type=str(data["primary_point_type"]),
            mask_area_px=int(data["mask_area_px"]),
            base_xy_m=tuple(float(v) for v in data["base_xy_m"]),
            request_id=str(data.get("request_id", "")),
            debug_image_url=data.get("debug_image_url"),
            created_at_unix_s=float(data.get("created_at_unix_s", time.time())),
        )


class GlobalDestinationLocator:
    """Small, self-contained client for the already deployed Grounded-SAM2 service.

    It intentionally reuses the same contract as robot_host_stage2a:
      - POST /api/v1/detect_segment
      - UNDISTORTED_1280x720 image space
      - mask centroid -> calibrated image-to-base homography

    No wrist depth is used for the destination in the fast tabletop demo. This is
    especially useful for transparent receptacles such as a glass bowl, for which
    D435i depth can be unreliable.
    """

    def __init__(
        self,
        base_url: str,
        homography_npy_path: str,
        expected_image_space_id: str,
        expected_width: int,
        expected_height: int,
        connect_timeout_s: float,
        read_timeout_s: float,
        min_confidence: float,
        min_mask_area_px: int,
        reject_ambiguous: bool,
        ambiguity_margin: float,
        workspace: Dict[str, float],
    ):
        self.base_url = str(base_url).rstrip("/")
        self.expected_image_space_id = str(expected_image_space_id)
        self.expected_width = int(expected_width)
        self.expected_height = int(expected_height)
        self.timeout_s = max(float(connect_timeout_s), float(read_timeout_s))
        self.min_confidence = float(min_confidence)
        self.min_mask_area_px = int(min_mask_area_px)
        self.reject_ambiguous = bool(reject_ambiguous)
        self.ambiguity_margin = float(ambiguity_margin)
        self.workspace = dict(workspace)

        p = Path(homography_npy_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Homography npy not found: {p}")
        H = np.asarray(np.load(str(p)), dtype=np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            raise ValueError("Homography must be a finite 3x3 matrix")
        if abs(float(np.linalg.det(H))) < 1e-12:
            raise ValueError("Homography is singular")
        self.H = H

    def _post_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"A6000 HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"A6000 request failed: {exc}") from exc
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError("A6000 response is not a JSON object")
        return data

    @staticmethod
    def _point2(value: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        return float(value[0]), float(value[1])

    def _project(self, uv: Sequence[float], image_space_id: str) -> Tuple[float, float]:
        if image_space_id != self.expected_image_space_id:
            raise RuntimeError(
                f"Homography/image-space mismatch: got={image_space_id}, "
                f"expected={self.expected_image_space_id}"
            )
        q = self.H @ np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64)
        if abs(float(q[2])) < 1e-10:
            raise RuntimeError("Invalid homography denominator")
        x = float(q[0] / q[2])
        y = float(q[1] / q[2])
        if not math.isfinite(x) or not math.isfinite(y):
            raise RuntimeError("Non-finite destination base XY")

        checks = (
            ("x_min", x, lambda a, b: a < b),
            ("x_max", x, lambda a, b: a > b),
            ("y_min", y, lambda a, b: a < b),
            ("y_max", y, lambda a, b: a > b),
        )
        for key, value, bad in checks:
            if key in self.workspace and bad(value, float(self.workspace[key])):
                raise RuntimeError(
                    f"Destination violates workspace {key}: value={value:.4f}, "
                    f"limit={float(self.workspace[key]):.4f}"
                )
        return x, y

    def locate(self, query: str, save_debug_image: bool = True) -> DestinationObservation:
        query = str(query).strip()
        if not query:
            raise ValueError("destination_query is empty")
        data = self._post_json(
            "/api/v1/detect_segment",
            {
                "text_prompt": query,
                "include_mask_rle": False,
                "save_debug_image": bool(save_debug_image),
            },
        )

        for key in (
            "request_id",
            "query",
            "frame_id",
            "timestamp",
            "width",
            "height",
            "image_space_id",
            "detections",
        ):
            if key not in data:
                raise RuntimeError(f"A6000 response missing field: {key}")

        width = int(data["width"])
        height = int(data["height"])
        image_space_id = str(data["image_space_id"])
        if (width, height) != (self.expected_width, self.expected_height):
            raise RuntimeError(
                f"Destination image resolution mismatch: {width}x{height}, "
                f"expected={self.expected_width}x{self.expected_height}"
            )
        if image_space_id != self.expected_image_space_id:
            raise RuntimeError(
                f"Destination image_space_id={image_space_id}, "
                f"expected={self.expected_image_space_id}"
            )

        candidates: List[Dict[str, Any]] = []
        for det in data.get("detections", []):
            try:
                conf = float(det.get("confidence", 0.0))
                area = int(det.get("mask_area_px", 0))
            except Exception:
                continue
            if conf < self.min_confidence or area < self.min_mask_area_px:
                continue
            centroid = self._point2(det.get("centroid_uv"))
            primary = self._point2(det.get("primary_uv")) or centroid
            bbox = det.get("bbox_xyxy")
            if primary is None or centroid is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            candidates.append(det)

        if not candidates:
            raise RuntimeError(
                f"No valid destination detection for '{query}' after confidence/mask filtering"
            )
        candidates.sort(
            key=lambda d: (
                float(d.get("confidence", 0.0)),
                float(d.get("sam_score", -1.0) if d.get("sam_score") is not None else -1.0),
                int(d.get("mask_area_px", 0)),
            ),
            reverse=True,
        )
        if self.reject_ambiguous and len(candidates) > 1:
            margin = float(candidates[0].get("confidence", 0.0)) - float(
                candidates[1].get("confidence", 0.0)
            )
            if margin < self.ambiguity_margin:
                raise RuntimeError(
                    f"Ambiguous destination '{query}': confidence margin={margin:.3f} "
                    f"< {self.ambiguity_margin:.3f}"
                )

        det = candidates[0]
        centroid = self._point2(det.get("centroid_uv"))
        primary = self._point2(det.get("primary_uv")) or centroid
        assert centroid is not None and primary is not None
        xy = self._project(primary, image_space_id)
        debug = data.get("debug_image_url")
        if debug and str(debug).startswith("/"):
            debug = self.base_url + str(debug)

        return DestinationObservation(
            query=query,
            label=str(det.get("label", query)),
            confidence=float(det.get("confidence", 0.0)),
            sam_score=(None if det.get("sam_score") is None else float(det["sam_score"])),
            frame_id=int(data["frame_id"]),
            timestamp=float(data["timestamp"]),
            image_space_id=image_space_id,
            width=width,
            height=height,
            bbox_xyxy=tuple(float(v) for v in det["bbox_xyxy"]),
            centroid_uv=centroid,
            primary_uv=primary,
            primary_point_type=str(det.get("primary_point_type", "mask_centroid")),
            mask_area_px=int(det.get("mask_area_px", 0)),
            base_xy_m=xy,
            request_id=str(data.get("request_id", "")),
            debug_image_url=(None if debug is None else str(debug)),
            created_at_unix_s=time.time(),
        )
