#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import atexit
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify

from camera_backend import create_camera_backend


def load_config(path: str):
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must contain a YAML mapping")
    return cfg, config_path


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

cfg, config_path = load_config(args.config)

camera_cfg = dict(cfg.get("camera", {}))
image_cfg = dict(cfg.get("image", {}))
server_cfg = dict(cfg.get("server", {}))
runtime_cfg = dict(cfg.get("runtime", {}))

intrinsic_file = Path(str(camera_cfg.get("intrinsic_file", "camera_intrinsics.npz")))
if not intrinsic_file.is_absolute():
    intrinsic_file = config_path.parent / intrinsic_file
camera_cfg["intrinsic_file"] = str(intrinsic_file)

WIDTH = int(camera_cfg.get("width", 1280))
HEIGHT = int(camera_cfg.get("height", 720))
IMAGE_SPACE_ID = str(image_cfg.get("image_space_id", "UNDISTORTED_1280x720"))
JPEG_QUALITY = int(image_cfg.get("jpeg_quality", 90))
HOST = str(server_cfg.get("host", "0.0.0.0"))
PORT = int(server_cfg.get("port", 8080))
CAPTURE_ERROR_SLEEP_S = float(runtime_cfg.get("capture_error_sleep_s", 0.2))

if not 1 <= JPEG_QUALITY <= 100:
    raise ValueError("image.jpeg_quality must be in [1, 100]")

app = Flask(__name__)
camera = create_camera_backend(camera_cfg)

frame_lock = threading.Lock()
latest_frame: Optional[np.ndarray] = None
latest_frame_id = 0
latest_timestamp = 0.0
last_capture_error = ""
camera_ready = False
stop_event = threading.Event()


def capture_loop():
    global latest_frame, latest_frame_id, latest_timestamp
    global last_capture_error, camera_ready

    try:
        camera.open()
        camera_ready = True
    except Exception as exc:
        last_capture_error = repr(exc)
        camera_ready = False
        print(f"[Capture] Fatal camera open error: {exc}")
        return

    print("[Capture] Background capture loop started")

    while not stop_event.is_set():
        try:
            frame = camera.read()
            ts = time.time()
            with frame_lock:
                latest_frame = frame
                latest_frame_id += 1
                latest_timestamp = ts
                last_capture_error = ""
        except Exception as exc:
            last_capture_error = repr(exc)
            print(f"[Capture] Read error: {exc}")
            time.sleep(CAPTURE_ERROR_SLEEP_S)


def cleanup():
    stop_event.set()
    try:
        camera.close()
    except Exception as exc:
        print(f"[Shutdown] camera.close() failed: {exc}")


atexit.register(cleanup)


@app.get("/health")
def health():
    with frame_lock:
        has_frame = latest_frame is not None
        frame_id = latest_frame_id
        timestamp = latest_timestamp
        error = last_capture_error

    ok = camera_ready and has_frame and not error

    return jsonify({
        "status": "ok" if ok else "waiting",
        "camera_ready": bool(camera_ready),
        "has_frame": bool(has_frame),
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "width": WIDTH,
        "height": HEIGHT,
        "encoding": "jpeg",
        "image_space_id": IMAGE_SPACE_ID,
        "do_undistort": bool(camera_cfg.get("do_undistort", True)),
        "last_error": error,
    }), 200 if ok else 503


@app.get("/api/v1/frame/latest/meta")
def latest_meta():
    with frame_lock:
        if latest_frame is None:
            return jsonify({
                "error": "camera frame not ready",
                "last_error": last_capture_error,
            }), 503

        frame_id = latest_frame_id
        timestamp = latest_timestamp

    return jsonify({
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "width": WIDTH,
        "height": HEIGHT,
        "encoding": "jpeg",
        "image_space_id": IMAGE_SPACE_ID,
    })


@app.get("/api/v1/frame/latest.jpg")
def latest_jpeg():
    with frame_lock:
        if latest_frame is None:
            return jsonify({
                "error": "camera frame not ready",
                "last_error": last_capture_error,
            }), 503

        frame = latest_frame.copy()
        frame_id = latest_frame_id
        timestamp = latest_timestamp

    h, w = frame.shape[:2]
    if w != WIDTH or h != HEIGHT:
        return jsonify({
            "error": "internal frame size mismatch",
            "expected": [WIDTH, HEIGHT],
            "actual": [w, h],
        }), 500

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        return jsonify({"error": "jpeg encode failed"}), 500

    response = Response(encoded.tobytes(), status=200, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Id"] = str(frame_id)
    response.headers["X-Timestamp"] = f"{timestamp:.6f}"
    response.headers["X-Image-Width"] = str(WIDTH)
    response.headers["X-Image-Height"] = str(HEIGHT)
    response.headers["X-Image-Space"] = IMAGE_SPACE_ID
    response.headers["X-Encoding"] = "jpeg"
    return response


if __name__ == "__main__":
    print("=" * 72)
    print("RK3588 Camera Service")
    print(f"Config           : {config_path}")
    print(f"Image space      : {IMAGE_SPACE_ID}")
    print(f"Output resolution: {WIDTH}x{HEIGHT}")
    print(f"JPEG quality     : {JPEG_QUALITY}")
    print(f"Listen           : http://{HOST}:{PORT}")
    print("=" * 72)

    capture_thread = threading.Thread(
        target=capture_loop,
        daemon=True,
        name="camera-capture",
    )
    capture_thread.start()

    app.run(
        host=HOST,
        port=PORT,
        threaded=True,
        debug=False,
        use_reloader=False,
    )
