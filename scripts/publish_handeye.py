#!/usr/bin/env python3
"""Publish only an explicitly confirmed, measured wrist-to-optical transform."""
import argparse
import math
import subprocess
from pathlib import Path

import yaml


def publisher_command(config):
    if config.get("calibrated") is not True:
        raise ValueError("Set calibrated=true only after completing hand-eye calibration")
    translation = config["translation_m"]
    quaternion = config["quaternion_xyzw"]
    if len(translation) != 3 or len(quaternion) != 4:
        raise ValueError("Expected 3 translation components and 4 quaternion components")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in translation + quaternion):
        raise ValueError("The calibration must contain finite measured numbers")
    if abs(sum(x * x for x in quaternion) - 1.0) > 1e-3:
        raise ValueError("Expected a unit quaternion in xyzw order")
    parent, child = config["parent_frame"], config["child_frame"]
    if not isinstance(parent, str) or not parent.strip() or not isinstance(child, str) or not child.strip() or parent == child:
        raise ValueError("Expected two different non-empty TF frame names")
    args = ["ros2", "run", "tf2_ros", "static_transform_publisher"]
    for key, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), translation + quaternion):
        args += ["--" + key, str(value)]
    return args + ["--frame-id", parent, "--child-frame-id", child]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    return subprocess.call(publisher_command(config))


if __name__ == "__main__":
    raise SystemExit(main())
