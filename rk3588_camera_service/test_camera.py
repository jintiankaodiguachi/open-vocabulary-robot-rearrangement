#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import yaml

from camera_backend import create_camera_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="test_undistorted.jpg")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    camera_cfg = dict(cfg["camera"])
    intrinsic_file = Path(str(camera_cfg.get("intrinsic_file", "camera_intrinsics.npz")))
    if not intrinsic_file.is_absolute():
        intrinsic_file = config_path.parent / intrinsic_file
    camera_cfg["intrinsic_file"] = str(intrinsic_file)

    camera = create_camera_backend(camera_cfg)

    try:
        camera.open()
        frame = camera.read()

        print(f"[TEST] frame.shape = {frame.shape}")
        print(f"[TEST] frame.dtype = {frame.dtype}")

        expected_shape = (
            int(camera_cfg["height"]),
            int(camera_cfg["width"]),
            3,
        )
        if frame.shape != expected_shape:
            raise RuntimeError(
                f"Expected shape {expected_shape}, got {frame.shape}"
            )

        ok = cv2.imwrite(args.output, frame)
        if not ok:
            raise RuntimeError(f"Failed to save {args.output}")

        print(f"[TEST] PASS. Saved: {Path(args.output).resolve()}")

    finally:
        camera.close()


if __name__ == "__main__":
    main()
