#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import threading
import time
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image
from torchvision.ops import box_convert

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class EngineConfig:
    repo_root: str
    sam2_checkpoint: str
    sam2_model_config: str
    grounding_dino_config: str
    grounding_dino_checkpoint: str
    device: str = "cuda"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    multimask_output: bool = False

    # Backward-compatible field. Existing app.py/offline_test.py can keep passing it.
    support_band_ratio: float = 0.03

    enable_tf32: bool = True

    # New robust support-point parameters.
    # These have defaults, so your existing app.py/offline_test.py do not need changes.
    support_x_band_ratio: float = 0.12
    support_y_quantile: float = 0.98
    support_min_band_px: int = 6


def _abs(repo_root: Path, path_str: str) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = repo_root / p
    return str(p.resolve())


def normalize_prompt(text: str) -> str:
    text = str(text).strip().lower()
    if not text:
        raise ValueError("text_prompt must not be empty")
    if not text.endswith("."):
        text += "."
    return text


def preprocess_grounding_image(image_bgr: np.ndarray) -> torch.Tensor:
    """
    GroundingDINO preprocessing.
    The model may resize internally, but all returned boxes are mapped back to
    the ORIGINAL input image size before SAM2 / JSON output.
    """
    # The external repo root is added during engine initialization.
    import grounding_dino.groundingdino.datasets.transforms as T

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tensor, _ = transform(image_pil, None)
    return image_tensor


def mask_centroid(mask: np.ndarray) -> Optional[List[float]]:
    """Mask centroid. This is the PRIMARY point for Stage 2A coarse localization."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    return [
        float(xs.mean()),
        float(ys.mean()),
    ]


def stable_support_point(
    mask: np.ndarray,
    x_band_ratio: float = 0.12,
    y_quantile: float = 0.98,
    min_band_px: int = 6,
) -> Optional[List[float]]:
    """
    Robust lower support/contact proxy.

    Old implementation:
      globally select the lowest few percent of the mask.

    Problem:
      for a slanted banana / scissors / elongated object, the globally lowest
      pixels often belong to one extreme tip, so the purple point jumps to a tip.

    New implementation:
      1) compute centroid x = cx
      2) build a vertical band around cx
      3) only look at mask pixels in that band
      4) take a high y quantile in that band
      5) return the median of the lower-boundary pixels

    This gives a bottom-center-like point rather than a global bottom-most tip.

    Important:
      support_uv is only an auxiliary 2-D proxy. For flat/lying objects such as
      banana, apple, tools, boxes, etc., centroid_uv remains the default point.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    cx = float(xs.mean())

    x_min = int(xs.min())
    x_max = int(xs.max())
    bbox_width = max(1, x_max - x_min + 1)

    half_band = max(
        int(min_band_px),
        int(round(bbox_width * float(x_band_ratio) * 0.5)),
    )

    in_band = np.abs(xs.astype(np.float32) - cx) <= half_band
    band_xs = xs[in_band]
    band_ys = ys[in_band]

    # Fallback for fragmented masks.
    if len(band_xs) < 10:
        band_xs = xs
        band_ys = ys

    q = float(np.clip(y_quantile, 0.50, 1.0))
    y_anchor = float(np.quantile(band_ys, q))

    # Use a small strip instead of one single extreme row.
    lower = band_ys >= (y_anchor - 2.0)
    lower_xs = band_xs[lower]
    lower_ys = band_ys[lower]

    if len(lower_xs) == 0:
        return [cx, y_anchor]

    return [
        float(np.median(lower_xs)),
        float(np.median(lower_ys)),
    ]


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_util.encode(
        np.asfortranarray(mask.astype(np.uint8))
    )

    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    return {
        "size": [
            int(rle["size"][0]),
            int(rle["size"][1]),
        ],
        "counts": rle["counts"],
    }


def draw_debug(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    masks: List[np.ndarray],
) -> np.ndarray:
    out = image_bgr.copy()
    overlay = out.copy()

    # Green SAM2 mask overlay.
    for mask in masks:
        overlay[mask] = (
            0.45 * overlay[mask]
            + 0.55 * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)

    out = cv2.addWeighted(
        overlay,
        0.55,
        out,
        0.45,
        0.0,
    )

    for det in detections:
        x1, y1, x2, y2 = [
            int(round(v)) for v in det["bbox_xyxy"]
        ]

        # Yellow bbox.
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            out,
            f'{det["label"]} {det["confidence"]:.2f}',
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # RED C = centroid = PRIMARY point.
        c = det.get("centroid_uv")
        if c is not None:
            cu = int(round(c[0]))
            cv = int(round(c[1]))

            cv2.circle(
                out,
                (cu, cv),
                5,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                out,
                "C",
                (cu + 7, cv - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # PURPLE S = secondary support/contact proxy.
        s = det.get("support_uv")
        if s is not None:
            su = int(round(s[0]))
            sv = int(round(s[1]))

            cv2.circle(
                out,
                (su, sv),
                6,
                (255, 0, 255),
                -1,
            )

            cv2.putText(
                out,
                "S",
                (su + 7, sv - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # Legend.
    cv2.putText(
        out,
        "C(red)=primary centroid",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "S(purple)=support proxy",
        (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return out


class GroundedSAM2Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.repo_root = Path(cfg.repo_root).resolve()

        if not self.repo_root.is_dir():
            raise FileNotFoundError(self.repo_root)
        # Grounded-SAM-2's bundled GroundingDINO uses this nested namespace
        # internally. Resolve it from the configured external checkout.
        repo_path = str(self.repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from grounding_dino.groundingdino.util.inference import load_model, predict

        self._grounding_predict = predict

        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False"
            )

        self.device = cfg.device

        if (
            self.device.startswith("cuda")
            and cfg.enable_tf32
            and torch.cuda.get_device_properties(0).major >= 8
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam_ckpt = _abs(self.repo_root, cfg.sam2_checkpoint)
        gd_cfg = _abs(self.repo_root, cfg.grounding_dino_config)
        gd_ckpt = _abs(self.repo_root, cfg.grounding_dino_checkpoint)

        for path in [sam_ckpt, gd_cfg, gd_ckpt]:
            if not Path(path).exists():
                raise FileNotFoundError(path)

        print("[AI] Loading SAM2...")

        sam2_model = build_sam2(
            cfg.sam2_model_config,
            sam_ckpt,
            device=self.device,
        )

        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        print("[AI] Loading GroundingDINO...")

        self.grounding_model = load_model(
            model_config_path=gd_cfg,
            model_checkpoint_path=gd_ckpt,
            device=self.device,
        )

        # SAM2 predictor stores the current image internally.
        self._lock = threading.Lock()

        print("[AI] Models ready")

        if self.device.startswith("cuda"):
            print("[AI] GPU:", torch.cuda.get_device_name(0))

    @torch.inference_mode()
    def infer(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
        include_mask_rle: bool = True,
        make_debug_image: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        if image_bgr is None:
            raise ValueError("image_bgr is None")

        h, w = image_bgr.shape[:2]
        prompt = normalize_prompt(text_prompt)

        with self._lock:
            total_start = time.perf_counter()

            image_tensor = preprocess_grounding_image(image_bgr)

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            grounding_start = time.perf_counter()

            boxes, confidences, labels = self._grounding_predict(
                model=self.grounding_model,
                image=image_tensor,
                caption=prompt,
                box_threshold=float(self.cfg.box_threshold),
                text_threshold=float(self.cfg.text_threshold),
                device=self.device,
            )

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            grounding_end = time.perf_counter()
            n = int(boxes.shape[0])

            if n == 0:
                result = {
                    "query": text_prompt,
                    "normalized_prompt": prompt,
                    "image_width": int(w),
                    "image_height": int(h),
                    "primary_point_type": "mask_centroid",
                    "detections": [],
                    "latency_ms": {
                        "grounding_ms":
                            (grounding_end - grounding_start) * 1000.0,
                        "sam_ms": 0.0,
                        "postprocess_ms": 0.0,
                        "total_model_ms":
                            (time.perf_counter() - total_start) * 1000.0,
                    },
                }

                debug = image_bgr.copy() if make_debug_image else None
                return result, debug

            # GroundingDINO normalized cxcywh -> original-image pixel xyxy.
            boxes_px_cxcywh = boxes * torch.tensor(
                [w, h, w, h],
                dtype=boxes.dtype,
                device=boxes.device,
            )

            input_boxes = box_convert(
                boxes=boxes_px_cxcywh,
                in_fmt="cxcywh",
                out_fmt="xyxy",
            ).detach().cpu().numpy()

            # SAM2 uses the original-resolution RGB image.
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.sam2_predictor.set_image(image_rgb)

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            sam_start = time.perf_counter()

            autocast_ctx = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                )
                if self.device.startswith("cuda")
                else nullcontext()
            )

            with autocast_ctx:
                masks, sam_scores, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=bool(self.cfg.multimask_output),
                )

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            sam_end = time.perf_counter()

            masks_np = np.asarray(masks)
            sam_scores_np = np.asarray(sam_scores)

            if self.cfg.multimask_output and masks_np.ndim == 4:
                best = np.argmax(sam_scores_np, axis=1)
                masks_np = masks_np[np.arange(masks_np.shape[0]), best]
                sam_scores_np = sam_scores_np[
                    np.arange(sam_scores_np.shape[0]), best
                ]
            elif masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0]

            if sam_scores_np.ndim > 1:
                sam_scores_np = sam_scores_np.reshape(
                    sam_scores_np.shape[0], -1
                )[:, 0]

            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]

            if torch.is_tensor(confidences):
                confidence_values = (
                    confidences.detach().cpu().numpy().tolist()
                )
            else:
                confidence_values = list(confidences)

            post_start = time.perf_counter()

            detections: List[Dict[str, Any]] = []
            debug_masks: List[np.ndarray] = []

            for i in range(n):
                mask = masks_np[i].astype(bool)

                c = mask_centroid(mask)

                s = stable_support_point(
                    mask,
                    x_band_ratio=float(self.cfg.support_x_band_ratio),
                    y_quantile=float(self.cfg.support_y_quantile),
                    min_band_px=int(self.cfg.support_min_band_px),
                )

                det: Dict[str, Any] = {
                    "id": int(i),
                    "label": str(labels[i]),
                    "confidence": float(confidence_values[i]),
                    "sam_score": float(np.asarray(sam_scores_np[i]).squeeze()),
                    "bbox_xyxy": [
                        float(v) for v in input_boxes[i].tolist()
                    ],
                    "centroid_uv": c,
                    "support_uv": s,

                    # IMPORTANT: Robot Host should use this by default.
                    "primary_uv": c,
                    "primary_point_type": "mask_centroid",

                    "mask_area_px": int(mask.sum()),
                }

                if include_mask_rle:
                    det["mask_rle"] = mask_to_rle(mask)

                detections.append(det)

                if make_debug_image:
                    debug_masks.append(mask)

            debug_image = None
            if make_debug_image:
                debug_image = draw_debug(
                    image_bgr,
                    detections,
                    debug_masks,
                )

            post_end = time.perf_counter()

            result = {
                "query": text_prompt,
                "normalized_prompt": prompt,
                "image_width": int(w),
                "image_height": int(h),
                "primary_point_type": "mask_centroid",
                "detections": detections,
                "latency_ms": {
                    "grounding_ms":
                        (grounding_end - grounding_start) * 1000.0,
                    "sam_ms":
                        (sam_end - sam_start) * 1000.0,
                    "postprocess_ms":
                        (post_end - post_start) * 1000.0,
                    "total_model_ms":
                        (time.perf_counter() - total_start) * 1000.0,
                },
            }

            return result, debug_image
