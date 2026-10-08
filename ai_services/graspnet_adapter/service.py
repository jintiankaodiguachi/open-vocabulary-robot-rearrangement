#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import uuid
from pathlib import Path

import yaml
from flask import Flask, jsonify, request

from graspnet_engine import GraspNetEngine


def create_app(config_path):
    config_path = Path(config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    engine = GraspNetEngine(str(config_path))
    app = Flask(__name__)
    lock = threading.Lock()

    output_root = Path(cfg["server"]["output_dir"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "graspnet_v6_official_scene_input",
            "device": str(engine.device),
            "checkpoint": str(engine.checkpoint_path),
        })

    @app.post("/api/v1/predict_grasps")
    def predict():
        if "npz" not in request.files:
            return jsonify({"error": "multipart field 'npz' is required"}), 400

        request_id = str(uuid.uuid4())
        req_dir = output_root / request_id
        req_dir.mkdir(parents=True, exist_ok=False)

        input_path = req_dir / "input.npz"
        request.files["npz"].save(str(input_path))

        try:
            with lock:
                _, result = engine.predict_npz(
                    str(input_path),
                    save_dir=str(req_dir),
                )

            result["request_id"] = request_id

            with open(
                output_root / "latest_result.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return jsonify(result)

        except Exception as exc:
            app.logger.exception("GraspNet V6 failed")
            return jsonify({
                "request_id": request_id,
                "error": type(exc).__name__,
                "message": str(exc),
            }), 500

    return app, cfg


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    app, cfg = create_app(args.config)
    app.run(
        host=cfg["server"]["host"],
        port=int(cfg["server"]["port"]),
        threaded=True,
    )
