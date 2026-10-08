#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup


STAGE_FILES = {
    "raw":
        "grasps_raw_camera.npy",
    "nms":
        "grasps_nms_camera.npy",
    "width":
        "grasps_width_camera.npy",
    "target":
        "grasps_target_camera.npy",
    "collision":
        "grasps_collision_free_camera.npy",
    "final":
        "grasps_final_camera.npy",
}


def _rgb01(colors):
    if colors is None:
        return None

    arr = np.asarray(
        colors
    )

    if arr.ndim != 2 or arr.shape[1] != 3:
        return None

    arr = arr.astype(
        np.float64
    )

    if (
        arr.size > 0
        and float(
            np.nanmax(arr)
        ) > 1.5
    ):
        arr = arr / 255.0

    return np.clip(
        arr,
        0.0,
        1.0,
    )


def make_colored_pcd(
    points,
    colors=None,
    fallback_color=(0.65, 0.65, 0.65),
):
    pts = np.asarray(
        points,
        dtype=np.float64,
    )

    pcd = (
        o3d.geometry.PointCloud()
    )

    pcd.points = (
        o3d.utility.Vector3dVector(
            pts
        )
    )

    rgb = _rgb01(
        colors
    )

    if (
        rgb is not None
        and len(rgb) == len(pts)
    ):
        pcd.colors = (
            o3d.utility.Vector3dVector(
                rgb
            )
        )
    else:
        pcd.paint_uniform_color(
            list(fallback_color)
        )

    return pcd


def recolor_gripper(
    geometry,
    color,
):
    try:
        geometry.paint_uniform_color(
            color
        )
    except Exception:
        pass

    return geometry


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "output_dir"
    )

    ap.add_argument(
        "--stage",
        choices=list(
            STAGE_FILES.keys()
        ),
        default="final",
        help=(
            "Visualize grasp candidates at a "
            "specific processing stage."
        ),
    )

    ap.add_argument(
        "--top",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--point-size",
        type=float,
        default=2.0,
    )

    ap.add_argument(
        "--no-object-overlay",
        action="store_true",
        help=(
            "Show only the true-color scene cloud. "
            "Useful because the target already exists "
            "inside the scene cloud."
        ),
    )

    args = ap.parse_args()

    out = Path(
        args.output_dir
    ).resolve()

    debug_path = (
        out
        / "grasp_debug_scene.npz"
    )

    grasp_path = (
        out
        / STAGE_FILES[
            args.stage
        ]
    )

    if not debug_path.exists():
        raise FileNotFoundError(
            debug_path
        )

    if not grasp_path.exists():
        raise FileNotFoundError(
            grasp_path
        )

    d = np.load(
        debug_path,
        allow_pickle=False,
    )

    scene_base = d[
        "scene_points_base"
    ]

    object_base = d[
        "object_points_base"
    ]

    T_base_camera = d[
        "T_base_camera"
    ]

    scene_rgb = (
        d["scene_colors_rgb"]
        if "scene_colors_rgb"
        in d.files
        else None
    )

    object_rgb = (
        d["object_colors_rgb"]
        if "object_colors_rgb"
        in d.files
        else None
    )

    if scene_rgb is None:
        print(
            "[WARN] scene_colors_rgb missing. "
            "The input NPZ was probably generated "
            "before RGB preservation was enabled."
        )
    else:
        print(
            "[RGB] scene color points:",
            len(scene_rgb),
        )

    if object_rgb is None:
        print(
            "[WARN] object_colors_rgb missing."
        )
    else:
        print(
            "[RGB] object color points:",
            len(object_rgb),
        )

    scene_pcd = make_colored_pcd(
        scene_base,
        scene_rgb,
        fallback_color=(
            0.65,
            0.65,
            0.65,
        ),
    )

    object_pcd = make_colored_pcd(
        object_base,
        object_rgb,
        fallback_color=(
            0.10,
            0.90,
            0.10,
        ),
    )

    gg_camera = GraspGroup(
        str(
            grasp_path
        )
    )

    # IMPORTANT:
    # Saved stage arrays may not have been sorted yet.
    # For visualization we explicitly display the highest
    # ORIGINAL GraspNet scores first.
    if len(gg_camera) > 0:
        order = np.argsort(
            gg_camera.scores.astype(
                np.float64
            )
        )[::-1]

        gg_camera = (
            gg_camera[order]
        )

    if len(gg_camera) > args.top:
        gg_camera = (
            gg_camera[
                :args.top
            ]
        )

    gg_base = GraspGroup(
        gg_camera.grasp_group_array.copy()
    )

    gg_base.transform(
        T_base_camera
    )

    grippers = []

    # TOP-1 = orange, TOP-2 = yellow, rest = blue.
    for i in range(
        len(gg_base)
    ):
        one = GraspGroup(
            gg_base[
                i:i + 1
            ].grasp_group_array.copy()
        )

        geom_list = (
            one.to_open3d_geometry_list()
        )

        if i == 0:
            color = [
                1.0,
                0.45,
                0.0,
            ]
        elif i == 1:
            color = [
                1.0,
                0.85,
                0.0,
            ]
        else:
            color = [
                0.10,
                0.25,
                1.0,
            ]

        for geom in geom_list:
            grippers.append(
                recolor_gripper(
                    geom,
                    color,
                )
            )

    frame = (
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=0.10,
            origin=[
                0.0,
                0.0,
                0.0,
            ],
        )
    )

    visualizer = (
        o3d.visualization.Visualizer()
    )

    visualizer.create_window(
        window_name=(
            "GraspNet V5 "
            f"[{args.stage}] — "
            "true RGB point cloud"
        ),
        width=1280,
        height=800,
    )

    visualizer.add_geometry(
        scene_pcd
    )

    # The target already exists in the RGB scene cloud.
    # Optional overlay makes sparse target geometry more visually obvious.
    if not args.no_object_overlay:
        visualizer.add_geometry(
            object_pcd
        )

    for g in grippers:
        visualizer.add_geometry(
            g
        )

    visualizer.add_geometry(
        frame
    )

    option = (
        visualizer.get_render_option()
    )

    option.background_color = (
        np.asarray(
            [1.0, 1.0, 1.0],
            dtype=np.float64,
        )
    )

    option.point_size = float(
        args.point_size
    )

    print(
        "\nStage:",
        args.stage,
    )

    print(
        "Displayed grasps:",
        len(gg_base),
    )

    for i in range(
        len(gg_camera)
    ):
        print(
            f"TOP {i + 1:02d} "
            f"score="
            f"{float(gg_camera.scores[i]):.6f} "
            f"width="
            f"{float(gg_camera.widths[i]):.4f}m"
        )

    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
