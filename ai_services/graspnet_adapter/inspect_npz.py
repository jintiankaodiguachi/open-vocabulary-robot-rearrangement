#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    args = ap.parse_args()

    d = np.load(
        args.npz,
        allow_pickle=False,
    )

    print("=" * 88)
    print("Stage 2B -> GraspNet V8 NPZ Contract")
    print("=" * 88)

    for key in d.files:
        a = d[key]
        print(
            f"{key:34s} "
            f"shape={str(a.shape):18s} "
            f"dtype={a.dtype}"
        )

    print()

    required = [
        "points_camera",
        "points_base",
        "dense_scene_points_camera",
        "dense_scene_points_base",
        "scene_points_camera",
        "scene_points_base",
        "T_base_camera",
        "table_z",
    ]

    missing = [
        k for k in required
        if k not in d.files
    ]

    if missing:
        print("[FAIL] missing:", missing)
        raise SystemExit(2)

    n_obj = len(d["points_camera"])
    n_dense = len(d["dense_scene_points_camera"])
    n_collision = len(d["scene_points_camera"])

    print(f"object points          : {n_obj}")
    print(f"dense GraspNet scene   : {n_dense}")
    print(f"collision/debug scene  : {n_collision}")

    if n_dense < 12000:
        print(
            "[WARN] dense scene is still small. "
            "Prefer >= 12000 unique valid RGB-D points; "
            ">= 20000 is ideal for this pipeline."
        )
    elif n_dense < 20000:
        print(
            "[OK/WARN] dense scene >= 12000 but < 20000. "
            "GraspNet will keep all unique points and "
            "sample only the remainder with replacement."
        )
    else:
        print(
            "[PASS] dense scene has >= 20000 unique points."
        )

    print("=" * 88)


if __name__ == "__main__":
    main()
