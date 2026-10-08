#!/usr/bin/env python3
"""
Persistent HTTP wrapper for the already-tested GraspNet V8 engine.

Design goals:
1. Load GraspNet once and keep it resident on the A6000.
2. Accept the local-perception NPZ generated on the robot host.
3. Run the existing V8 engine without changing its ranking/filter logic.
4. Return a stable, robot-facing JSON schema in base_link.
5. Be tolerant to small differences in V8 engine return/output naming.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import yaml
from flask import Flask, jsonify, request

from graspnet_engine import GraspNetEngine

GRASP_ARRAY_LEN = 17
ENGINE_LOCK = threading.Lock()


def _to_builtin(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _candidate_from_dict(c: dict, rank: int) -> Optional[dict]:
    def first(keys, default=None):
        for k in keys:
            if k in c:
                return c[k]
        return default

    p = first(("translation_base_grasp_m", "translation_base_m", "translation_base", "position_base_m"))
    R = first(("rotation_base_grasp", "rotation_base", "rotation_matrix_base"))
    if p is None or R is None:
        return None

    p = np.asarray(p, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    approach = np.asarray(first(("approach_axis_base",), R[:, 0]), dtype=np.float64).reshape(3)
    jaw = np.asarray(first(("jaw_axis_base",), R[:, 1]), dtype=np.float64).reshape(3)

    return {
        "rank": int(rank),
        "model_score": float(first(("model_score", "score", "grasp_score"), 0.0)),
        "width_m": float(first(("width_m", "width"), 0.0)),
        "height_m": float(first(("height_m", "height"), 0.02)),
        "depth_m": float(first(("depth_m", "depth"), 0.02)),
        "translation_base_grasp_m": p.tolist(),
        "rotation_base_grasp": R.tolist(),
        "approach_axis_base": approach.tolist(),
        "jaw_axis_base": jaw.tolist(),
        "approach_angle_from_down_deg": _to_builtin(first(("approach_angle_from_down_deg",), None)),
        "target_nearest_distance_m": _to_builtin(first(("target_nearest_distance_m", "target_dist_m"), None)),
    }


def _extract_candidate_list(result: Any) -> List[dict]:
    if result is None:
        return []
    if isinstance(result, tuple):
        for item in result:
            out = _extract_candidate_list(item)
            if out:
                return out
        return []
    if not isinstance(result, dict):
        return []

    for key in ("final_candidates", "final_grasps", "grasps", "candidates", "top_candidates"):
        value = result.get(key)
        if isinstance(value, list):
            out = []
            for i, c in enumerate(value):
                if isinstance(c, dict):
                    x = _candidate_from_dict(c, i)
                    if x is not None:
                        out.append(x)
            if out:
                out.sort(key=lambda x: x["model_score"], reverse=True)
                for i, c in enumerate(out):
                    c["rank"] = i
                return out

    stages = result.get("stages")
    if isinstance(stages, dict):
        for key in ("final", "collision", "target", "nms"):
            value = stages.get(key)
            if isinstance(value, list):
                out = []
                for i, c in enumerate(value):
                    if isinstance(c, dict):
                        x = _candidate_from_dict(c, i)
                        if x is not None:
                            out.append(x)
                if out:
                    out.sort(key=lambda x: x["model_score"], reverse=True)
                    for i, c in enumerate(out):
                        c["rank"] = i
                    return out

    top = result.get("top_grasp")
    if isinstance(top, dict):
        x = _candidate_from_dict(top, 0)
        return [x] if x else []
    return []


def _find_stage_npy(output_dir: Path) -> Optional[Path]:
    all_npy = sorted(output_dir.rglob("*.npy"))
    for token in ("final", "after_nms", "nms", "collision"):
        for p in all_npy:
            if token in p.name.lower():
                try:
                    arr = np.load(p, allow_pickle=False)
                    if arr.ndim == 2 and arr.shape[1] == GRASP_ARRAY_LEN:
                        return p
                except Exception:
                    pass
    return None


def _candidates_from_grasp_array(arr: np.ndarray, T_base_camera: np.ndarray) -> List[dict]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != GRASP_ARRAY_LEN:
        raise ValueError(f"Expected Nx17 GraspNet array, got {arr.shape}")

    R_bc = T_base_camera[:3, :3]
    t_bc = T_base_camera[:3, 3]
    out = []
    for row in arr:
        R_cg = row[4:13].reshape(3, 3)
        p_cg = row[13:16]
        R_bg = R_bc @ R_cg
        p_bg = R_bc @ p_cg + t_bc
        angle = float(np.degrees(np.arccos(np.clip(np.dot(R_bg[:, 0], [0.0, 0.0, -1.0]), -1.0, 1.0))))
        out.append({
            "rank": 0,
            "model_score": float(row[0]),
            "width_m": float(row[1]),
            "height_m": float(row[2]),
            "depth_m": float(row[3]),
            "translation_base_grasp_m": p_bg.tolist(),
            "rotation_base_grasp": R_bg.tolist(),
            "approach_axis_base": R_bg[:, 0].tolist(),
            "jaw_axis_base": R_bg[:, 1].tolist(),
            "approach_angle_from_down_deg": angle,
            "target_nearest_distance_m": None,
        })
    out.sort(key=lambda x: x["model_score"], reverse=True)
    for i, c in enumerate(out):
        c["rank"] = i
    return out


def _call_engine(engine: Any, npz_path: str, output_dir: str) -> Any:
    if not hasattr(engine, "predict_npz"):
        raise RuntimeError("GraspNetEngine must expose predict_npz(npz_path, output_dir=...).")
    fn = engine.predict_npz
    try:
        return fn(npz_path, output_dir=output_dir)
    except TypeError:
        return fn(npz_path, output_dir)


def create_app(config_path: str, runtime_dir: str) -> Flask:
    cfg = _load_yaml(config_path)
    runtime_root = Path(runtime_dir)
    runtime_root.mkdir(parents=True, exist_ok=True)

    print(f"[GraspService] loading engine with config: {config_path}", flush=True)
    engine = GraspNetEngine(config_path)
    print("[GraspService] engine ready", flush=True)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(cfg.get("service", {}).get("max_upload_mb", 64)) * 1024 * 1024

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "engine": "GraspNetEngine", "config": config_path})

    @app.post("/api/v1/grasp_npz")
    def grasp_npz():
        t0 = time.perf_counter()
        req_id = str(uuid.uuid4())
        if "file" not in request.files:
            return jsonify({"status": "error", "error": "multipart field 'file' is required"}), 400

        req_dir = runtime_root / req_id
        out_dir = req_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = req_dir / "input.npz"
        request.files["file"].save(npz_path)

        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "T_base_camera" not in data:
                    raise ValueError("NPZ missing T_base_camera")
                T_base_camera = np.asarray(data["T_base_camera"], dtype=np.float64).reshape(4, 4)
                label = str(data["label"].item()) if "label" in data else "unknown"
                frame_base = str(data["frame_base"].item()) if "frame_base" in data else "base_link"
                object_id = str(data["object_id"].item()) if "object_id" in data else ""
                table_z = float(data["table_z"]) if "table_z" in data else None

            with ENGINE_LOCK:
                engine_result = _call_engine(engine, str(npz_path), str(out_dir))

            candidates = _extract_candidate_list(engine_result)
            if not candidates:
                stage_npy = _find_stage_npy(out_dir)
                if stage_npy is not None:
                    candidates = _candidates_from_grasp_array(np.load(stage_npy, allow_pickle=False), T_base_camera)

            max_return = int(cfg.get("service", {}).get("max_return_candidates", 20))
            candidates = candidates[:max_return]
            response = {
                "status": "ok",
                "request_id": req_id,
                "label": label,
                "object_id": object_id,
                "frame_base": frame_base,
                "table_z_m": table_z,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "engine_result_summary": _to_builtin(engine_result) if isinstance(engine_result, dict) else None,
                "latency_ms": {"total": (time.perf_counter() - t0) * 1000.0},
            }
            if not candidates:
                response["warning"] = "No V8 final candidates were found."
            with open(req_dir / "response.json", "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            return jsonify(response)
        except Exception as exc:
            err = {"status": "error", "request_id": req_id, "error": f"{type(exc).__name__}: {exc}"}
            with open(req_dir / "error.json", "w", encoding="utf-8") as f:
                json.dump(err, f, indent=2, ensure_ascii=False)
            return jsonify(err), 500

    return app


def main():
    parser = argparse.ArgumentParser()
    service_dir = Path(__file__).resolve().parent
    parser.add_argument("--config", default=str(service_dir / "config.yaml"))
    parser.add_argument("--runtime-dir", default=str(service_dir / "runtime"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    create_app(args.config, args.runtime_dir).run(host=args.host, port=args.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()
