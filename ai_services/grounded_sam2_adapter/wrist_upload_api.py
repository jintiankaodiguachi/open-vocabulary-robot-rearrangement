#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Blueprint, jsonify, request


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def register_wrist_upload_api(app, engine, output_dir: Optional[Path] = None):
    """
    Add a wrist-image endpoint to the EXISTING Grounded-SAM2 Flask service.

    This reuses the already-loaded GroundingDINO + SAM2 models. It does NOT
    load a second copy of the models.

    Endpoint:
      POST /api/v1/segment_uploaded

    multipart/form-data:
      image: JPEG/PNG file
      text_prompt: e.g. banana
      include_mask_rle: true/false
      save_debug_image: true/false
    """
    bp = Blueprint("wrist_upload_api", __name__)

    out = None
    if output_dir is not None:
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

    @bp.post("/api/v1/segment_uploaded")
    def segment_uploaded():
        request_id = str(uuid.uuid4())

        try:
            if "image" not in request.files:
                return jsonify({
                    "error": "image file is required",
                    "request_id": request_id,
                }), 400

            prompt = str(request.form.get("text_prompt", "")).strip()
            if not prompt:
                return jsonify({
                    "error": "text_prompt is required",
                    "request_id": request_id,
                }), 400

            include_mask_rle = _as_bool(
                request.form.get("include_mask_rle"),
                True,
            )
            save_debug = _as_bool(
                request.form.get("save_debug_image"),
                False,
            )

            raw = request.files["image"].read()
            encoded = np.frombuffer(raw, dtype=np.uint8)

            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image_bgr is None:
                return jsonify({
                    "error": "failed to decode uploaded image",
                    "request_id": request_id,
                }), 400

            t0 = time.perf_counter()

            # GroundedSAM2Engine internally serializes access to SAM2 predictor,
            # so global-camera and wrist-camera HTTP calls are safe.
            result, debug_image = engine.infer(
                image_bgr=image_bgr,
                text_prompt=prompt,
                include_mask_rle=include_mask_rle,
                make_debug_image=save_debug,
            )

            debug_url = None
            if save_debug and debug_image is not None and out is not None:
                name = f"wrist_{request_id}_debug.jpg"
                cv2.imwrite(str(out / name), debug_image)
                cv2.imwrite(str(out / "latest_wrist_detection.jpg"), debug_image)
                debug_url = f"/outputs/{name}"

            result.update({
                "request_id": request_id,
                "source": "uploaded_wrist_rgb",
                "encoding": "jpeg_or_png_upload",
                "debug_image_url": debug_url,
            })

            result.setdefault("latency_ms", {})
            result["latency_ms"]["http_total_ms"] = (
                time.perf_counter() - t0
            ) * 1000.0

            if out is not None:
                with open(
                    out / "latest_wrist_result.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        result,
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

            return jsonify(result)

        except Exception as exc:
            app.logger.exception("segment_uploaded failed")
            return jsonify({
                "error": type(exc).__name__,
                "message": str(exc),
                "request_id": request_id,
            }), 500

    app.register_blueprint(bp)
