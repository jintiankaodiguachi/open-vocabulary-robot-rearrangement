#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup


STAGE_FILES = {
    "raw": "grasps_raw_camera.npy",
    "width": "grasps_width_camera.npy",
    "target": "grasps_target_camera.npy",
    "collision": "grasps_collision_free_camera.npy",
    "nms": "grasps_nms_camera.npy",
    "final": "grasps_final_camera.npy",
}


def make_pcd(points, color):
    cloud = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float64)
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.paint_uniform_color(color)
    return cloud


def recolor(geom, color):
    try:
        geom.paint_uniform_color(color)
    except Exception:
        pass
    return geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument(
        "--stage",
        choices=STAGE_FILES.keys(),
        default="final",
    )
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--point-size", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()

    debug = np.load(
        out / "grasp_debug_scene.npz",
        allow_pickle=False,
    )

    scene_base = debug["scene_points_base"]
    object_base = debug["object_points_base"]
    T_base_camera = debug["T_base_camera"]

    grasp_path = out / STAGE_FILES[args.stage]
    if not grasp_path.exists():
        raise FileNotFoundError(grasp_path)

    gg_cam = GraspGroup(str(grasp_path))

    if len(gg_cam):
        order = np.argsort(
            gg_cam.scores.astype(np.float64)
        )[::-1]
        gg_cam = gg_cam[order]

    if len(gg_cam) > args.top:
        gg_cam = gg_cam[:args.top]

    gg_base = GraspGroup(
        gg_cam.grasp_group_array.copy()
    )
    gg_base.transform(T_base_camera)

    scene = make_pcd(
        scene_base,
        [0.70, 0.70, 0.70],
    )
    target = make_pcd(
        object_base,
        [0.05, 0.90, 0.05],
    )

    geoms = [scene, target]

    for i in range(len(gg_base)):
        one = GraspGroup(
            gg_base[
                i:i + 1
            ].grasp_group_array.copy()
        )

        geometries = one.to_open3d_geometry_list()

        if i == 0:
            color = [1.0, 0.45, 0.0]
        elif i == 1:
            color = [1.0, 0.85, 0.0]
        else:
            color = [0.10, 0.20, 1.0]

        for g in geometries:
            geoms.append(
                recolor(g, color)
            )

    geoms.append(
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=0.10,
            origin=[0.0, 0.0, 0.0],
        )
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=(
            f"GraspNet V8 [{args.stage}] — "
            "gray collision scene + green target"
        ),
        width=1280,
        height=800,
    )

    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array(
        [1.0, 1.0, 1.0]
    )
    opt.point_size = float(
        args.point_size
    )

    print("Stage:", args.stage)
    print("Displayed grasps:", len(gg_cam))

    for i in range(len(gg_cam)):
        print(
            f"TOP {i+1:02d} "
            f"score={float(gg_cam.scores[i]):.6f} "
            f"width={float(gg_cam.widths[i]):.4f}m"
        )

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
