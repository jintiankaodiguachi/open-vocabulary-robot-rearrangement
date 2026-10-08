#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup


def cloud_to_o3d(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    if len(points) > 0:
        colors = np.tile(
            np.asarray(color, dtype=np.float64).reshape(1, 3),
            (len(points), 1),
        )
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "output_dir",
        help="Directory produced by offline_test.py",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=30,
    )
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()

    clouds = np.load(
        out / "grasp_debug_scene.npz",
        allow_pickle=False,
    )

    scene = clouds["scene_points_camera"]
    obj = clouds["object_points_camera"]

    gg = GraspGroup(
        str(out / "grasps_camera.npy")
    )

    if len(gg) > args.top:
        gg = gg[:args.top]

    scene_pcd = cloud_to_o3d(
        scene,
        [0.55, 0.55, 0.55],
    )
    object_pcd = cloud_to_o3d(
        obj,
        [0.1, 0.9, 0.1],
    )

    grippers = gg.to_open3d_geometry_list()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.10,
        origin=[0.0, 0.0, 0.0],
    )

    o3d.visualization.draw_geometries(
        [
            scene_pcd,
            object_pcd,
            frame,
            *grippers,
        ],
        window_name="GraspNet camera-frame candidates",
        width=1280,
        height=800,
    )


if __name__ == "__main__":
    main()
