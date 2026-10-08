#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def cloud(points, color):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    p.paint_uniform_color(color)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--point-size", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    d = np.load(out / "sampling_debug.npz", allow_pickle=False)

    sampled = d["sampled_scene_camera"]
    labels = d["sampled_scene_strata"].astype(np.uint8)
    target = d["object_points_camera"]

    result_path = out / "result.json"
    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        print(json.dumps(result.get("sampling", {}), indent=2))

    # Gray background, yellow/green local context, red core, cyan target cloud.
    geoms = []

    bg = sampled[labels == 2]
    ctx = sampled[labels == 1]
    core = sampled[labels == 0]

    if len(bg):
        geoms.append(cloud(bg, [0.70, 0.70, 0.70]))
    if len(ctx):
        geoms.append(cloud(ctx, [0.25, 0.75, 0.25]))
    if len(core):
        geoms.append(cloud(core, [1.00, 0.20, 0.10]))
    if len(target):
        geoms.append(cloud(target, [0.10, 0.30, 1.00]))

    geoms.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.08,
            origin=[0.0, 0.0, 0.0],
        )
    )

    print(f"sampled total : {len(sampled)}")
    print(f"core          : {len(core)}")
    print(f"context       : {len(ctx)}")
    print(f"background    : {len(bg)}")
    print(f"SAM2 target   : {len(target)}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="GraspNet V8 target-aware sampling",
        width=1280,
        height=800,
    )
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = float(args.point_size)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
