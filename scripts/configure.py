#!/usr/bin/env python3
"""Generate ignored per-machine configuration without enabling real execution."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from urllib.parse import urlsplit

import yaml

CHECKOUT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = "/opt/open-vocabulary-robot-rearrangement"
ROS_TEMPLATES = {
    "observer": ("open_vocab_observer", "observer.yaml"),
    "local": ("open_vocab_local_perception", "wrist_local_perception.yaml"),
    "grasp": ("open_vocab_grasp_planner", "grasp_planner.yaml"),
    "pick": ("open_vocab_manipulation", "manipulation.yaml"),
    "place": ("open_vocab_manipulation", "place.yaml"),
    "home": ("open_vocab_task_manager", "home.yaml"),
    "task": ("open_vocab_task_manager", "task_manager.yaml"),
}
EXECUTION_KEYS = {
    "execute_motion", "enable_arm_execution", "allow_approach_dry_run",
    "enable_gripper_execution", "enable_grasp_and_lift",
    "enable_place_arm_execution", "enable_release_execution", "enable_home_execution",
}


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def rewrite(value, replacements):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [rewrite(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, replacements) for key, item in value.items()}
    return value


def lock_execution(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in EXECUTION_KEYS:
                value[key] = False
            else:
                lock_execution(item)
    elif isinstance(value, list):
        for item in value:
            lock_execution(item)


def url(value):
    value = str(value).rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Expected a valid HTTP service URL: {value}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Service URLs must not contain credentials, query strings, or fragments")
    return value


def absolute(value, root: Path, fallback: Path):
    path = Path(str(value)).expanduser() if value else fallback
    return (path if path.is_absolute() else root / path).resolve().as_posix()


def generate(site, output: Path):
    root = Path(str(site.get("repo_root") or CHECKOUT)).expanduser().resolve()
    services = site["services"]
    sam = url(services["grounded_sam2_url"])
    grasp = url(services["graspnet_url"])
    camera = url(services["camera_url"])
    calibration = site["calibration"]
    models = site["models"]
    if not isinstance(calibration["geometry_confirmed"], bool) or not isinstance(calibration["homography_configured"], bool):
        raise ValueError("Calibration confirmations must be YAML booleans")
    table_z = calibration.get("table_z_m")
    if table_z is not None and (isinstance(table_z, bool) or not isinstance(table_z, (int, float)) or not math.isfinite(table_z)):
        raise ValueError("table_z_m must be a finite number or null")
    if calibration["geometry_confirmed"] and table_z is None:
        raise ValueError("geometry_confirmed=true requires a measured table_z_m")
    homography = absolute(calibration.get("homography_npy_path"), root, root / "perception/calibration/H_image_to_base.npy")
    intrinsics = absolute(calibration.get("camera_intrinsics_npz_path"), root, root / "rk3588_camera_service/camera_intrinsics.npz")
    replacements = {
        PUBLIC_ROOT: root.as_posix(),
        "http://ai-server.local:8090": sam,
        "http://ai-server.local:8091": grasp,
        "http://camera-host.local:8080": camera,
    }
    generated = {}
    for label, (package, filename) in ROS_TEMPLATES.items():
        data = rewrite(load(CHECKOUT / "ros2_ws/src" / package / "config" / filename), replacements)
        lock_execution(data)
        # All modules must use the same measured tabletop reference.
        for node in data.values():
            params = node.get("ros__parameters", {}) if isinstance(node, dict) else {}
            if table_z is not None:
                for key in ("table_z", "table_top_z_m", "table_physical_top_z_m"):
                    if key in params:
                        params[key] = float(table_z)
            if "homography_npy_path" in params:
                params["homography_npy_path"] = homography
        if label == "observer":
            data["open_vocab_observation_pose"]["ros__parameters"]["geometry_configured"] = calibration["geometry_confirmed"]
        if label == "task":
            params = data["open_vocab_task_manager"]["ros__parameters"]
            params["stage2a_config"] = (output / "robot_host.yaml").as_posix()
            params["stage2a_pass_config_arg"] = True
        generated[Path("ros") / filename] = data
    generated[Path("ros/destination_profiles.yaml")] = load(CHECKOUT / "ros2_ws/src/open_vocab_task_manager/config/destination_profiles.yaml")

    robot = rewrite(load(CHECKOUT / "perception/robot_host/config.yaml"), replacements)
    robot["homography"]["configured"] = calibration["homography_configured"]
    robot["homography"]["npy_path"] = homography
    robot["scene"]["output_dir"] = (root / "perception/robot_host/outputs").as_posix()
    generated[Path("robot_host.yaml")] = robot

    sam_config = rewrite(load(CHECKOUT / "ai_services/grounded_sam2_adapter/config.yaml"), replacements)
    sam_config["model"]["repo_root"] = absolute(models["grounded_sam2_repo_root"], root, root / "third_party/Grounded-SAM-2")
    for key in ("sam2_checkpoint", "grounding_dino_config", "grounding_dino_checkpoint"):
        sam_config["model"][key] = models[key]
    sam_config["camera"]["latest_jpeg_url"] = camera + "/api/v1/frame/latest.jpg"
    sam_config["output"]["dir"] = (root / "ai_services/grounded_sam2_adapter/outputs").as_posix()
    generated[Path("grounded_sam2.yaml")] = sam_config

    grasp_config = load(CHECKOUT / "ai_services/graspnet_adapter/config.yaml")
    grasp_config["graspnet"]["repo_root"] = absolute(models["graspnet_repo_root"], root, root / "third_party/graspnet-baseline")
    grasp_config["graspnet"]["checkpoint_path"] = absolute(models["graspnet_checkpoint"], root, root / "checkpoints/checkpoint-rs.tar")
    generated[Path("graspnet.yaml")] = grasp_config

    camera_config = load(CHECKOUT / "rk3588_camera_service/config.yaml")
    camera_config["camera"]["intrinsic_file"] = intrinsics
    camera_config["camera"]["device"] = calibration["camera_device"]
    generated[Path("camera.yaml")] = camera_config

    # Validate everything before writing. Nothing connects to the robot or servers.
    for relative, data in generated.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=CHECKOUT / "config/example.yaml")
    parser.add_argument("--output", type=Path, default=CHECKOUT / "config/local")
    args = parser.parse_args()
    generated = generate(load(args.site), args.output.expanduser().resolve())
    print(f"Generated {len(generated)} configuration files in {args.output.resolve()}")
    print("All execution switches are false; calibration and model installation are still required.")


if __name__ == "__main__":
    main()
