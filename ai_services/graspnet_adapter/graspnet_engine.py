#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree


class GraspNetEngine:
    """
    V8 data contract:

        dense_scene_points_camera
            -> pretrained GraspNet network input

        points_camera
            -> SAM2 target object cloud
            -> semantic target association

        scene_points_camera
            -> voxel-downsampled local collision cloud
            -> ModelFreeCollisionDetector

    Pipeline:
        dense local scene + SAM2 target
          -> target-aware stratified sample 20k
          -> GraspNet
          -> physical width
          -> target association
          -> model-free collision on sparse scene
          -> NMS
          -> ORIGINAL GraspNet model score
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        gcfg = self.cfg["graspnet"]

        self.repo_root = Path(gcfg["repo_root"]).resolve()
        self.checkpoint_path = Path(
            gcfg["checkpoint_path"]
        ).resolve()

        if not self.repo_root.exists():
            raise FileNotFoundError(self.repo_root)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)

        for rel in ("models", "dataset", "utils"):
            p = str(self.repo_root / rel)
            if p not in sys.path:
                sys.path.insert(0, p)

        from graspnet import GraspNet, pred_decode
        from collision_detector import ModelFreeCollisionDetector
        from graspnetAPI import GraspGroup

        self.GraspNet = GraspNet
        self.pred_decode = pred_decode
        self.ModelFreeCollisionDetector = ModelFreeCollisionDetector
        self.GraspGroup = GraspGroup

        requested = str(gcfg.get("device", "cuda:0"))
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {requested}, but CUDA is unavailable"
            )

        self.device = torch.device(requested)
        self.num_point = int(gcfg.get("num_point", 20000))
        self.rng = np.random.default_rng(
            int(gcfg.get("random_seed", 1234))
        )

        self.net = self.GraspNet(
            input_feature_dim=0,
            num_view=int(gcfg.get("num_view", 300)),
            num_angle=int(gcfg.get("num_angle", 12)),
            num_depth=int(gcfg.get("num_depth", 4)),
            cylinder_radius=float(
                gcfg.get("cylinder_radius", 0.05)
            ),
            hmin=float(gcfg.get("hmin", -0.02)),
            hmax_list=[
                float(v)
                for v in gcfg.get(
                    "hmax_list",
                    [0.01, 0.02, 0.03, 0.04],
                )
            ],
            is_training=False,
        ).to(self.device)

        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
        )

        self.net.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        self.net.eval()

        print(
            "[GraspNet] loaded "
            f"{self.checkpoint_path} "
            f"epoch={checkpoint.get('epoch', None)} "
            f"device={self.device}"
        )

    @staticmethod
    def _sanitize_points(points, positive_z=False):
        pts = np.asarray(
            points,
            dtype=np.float32,
        ).reshape(-1, 3)

        keep = np.isfinite(pts).all(axis=1)

        if positive_z:
            keep &= pts[:, 2] > 0.0

        return (
            np.ascontiguousarray(
                pts[keep],
                dtype=np.float32,
            ),
            keep,
        )

    def _sample_uniform(self, points):
        """
        Compatibility / A-B baseline sampler.

        Mirrors the old V7 behavior:
          N >= num_point: random sample without replacement
          N <  num_point: keep all unique points once, then replacement fill
        """
        pts = np.asarray(points, dtype=np.float32)
        n = len(pts)

        if n == 0:
            raise RuntimeError("Cannot sample empty dense scene")

        if n >= self.num_point:
            idx = self.rng.choice(n, self.num_point, replace=False)
        else:
            base = np.arange(n, dtype=np.int64)
            extra = self.rng.choice(
                n,
                self.num_point - n,
                replace=True,
            )
            idx = np.concatenate([base, extra], axis=0)
            self.rng.shuffle(idx)

        labels = np.full(len(idx), 2, dtype=np.uint8)
        stats = {
            "mode": "uniform",
            "dense_scene_points": int(n),
            "requested_points": int(self.num_point),
            "core_radius_m": None,
            "context_radius_m": None,
            "strata_available": {
                "core": 0,
                "context": 0,
                "background": int(n),
            },
            "strata_sampled": {
                "core": 0,
                "context": 0,
                "background": int(len(idx)),
            },
            "unique_sampled_points": int(len(np.unique(idx))),
            "replacement_points": int(len(idx) - len(np.unique(idx))),
        }

        return (
            np.ascontiguousarray(pts[idx], dtype=np.float32),
            np.ascontiguousarray(idx, dtype=np.int64),
            labels,
            stats,
        )

    def _sample_target_stratified(self, dense_scene, object_camera):
        """
        Target-aware stratified sampling for small open-vocabulary targets.

        IMPORTANT:
        This is NOT object-only inference and it does NOT fabricate/jitter points.
        When the dense scene has >= num_point points, every selected point is a
        unique point from the real aligned RGB-D scene.

        The dense scene is partitioned by 3-D nearest distance to the SAM2
        object cloud:

          CORE:
              object surface + immediate support/contact neighborhood
              distance <= core_radius_m

          CONTEXT:
              local tabletop / local geometry around the target
              core_radius_m < distance <= context_radius_m

          BACKGROUND:
              remaining scene context
              distance > context_radius_m

        Quotas bias the 20k network input toward CORE/CONTEXT while always
        preserving a configurable amount of BACKGROUND scene geometry.
        """
        pts = np.asarray(dense_scene, dtype=np.float32)
        target = np.asarray(object_camera, dtype=np.float32)

        n = len(pts)
        if n == 0:
            raise RuntimeError("Cannot sample empty dense scene")
        if len(target) == 0:
            raise RuntimeError("Cannot target-sample without object cloud")

        scfg = self.cfg.get("sampling", {})

        core_radius = float(scfg.get("core_radius_m", 0.018))
        context_radius = float(scfg.get("context_radius_m", 0.10))

        if core_radius <= 0.0:
            raise ValueError("sampling.core_radius_m must be > 0")
        if context_radius <= core_radius:
            raise ValueError(
                "sampling.context_radius_m must be > core_radius_m"
            )

        core_fraction = float(scfg.get("core_fraction", 0.40))
        context_fraction = float(scfg.get("context_fraction", 0.40))
        background_fraction = float(scfg.get("background_fraction", 0.20))

        fractions = np.array(
            [core_fraction, context_fraction, background_fraction],
            dtype=np.float64,
        )
        if np.any(fractions < 0.0):
            raise ValueError("sampling fractions must be non-negative")
        if float(np.sum(fractions)) <= 0.0:
            raise ValueError("sampling fractions sum must be > 0")
        fractions = fractions / np.sum(fractions)

        # Exact semantic geometry comes from SAM2 target cloud; use a KD-tree
        # only to define which REAL dense-scene points belong to each stratum.
        tree = cKDTree(target.astype(np.float64))
        nearest_dist, _ = tree.query(
            pts.astype(np.float64),
            k=1,
            workers=-1,
        )

        core_idx = np.flatnonzero(nearest_dist <= core_radius)
        context_idx = np.flatnonzero(
            (nearest_dist > core_radius)
            & (nearest_dist <= context_radius)
        )
        background_idx = np.flatnonzero(nearest_dist > context_radius)

        strata = [core_idx, context_idx, background_idx]
        names = ["core", "context", "background"]

        # Integer quotas that sum exactly to num_point.
        q_core = int(round(self.num_point * fractions[0]))
        q_context = int(round(self.num_point * fractions[1]))
        q_background = self.num_point - q_core - q_context
        quotas = [q_core, q_context, q_background]

        selected_parts = []
        selected_label_parts = []
        selected_set = set()
        sampled_counts = {name: 0 for name in names}

        # First pass: satisfy each stratum's requested quota WITHOUT replacement.
        for label_id, (name, pool, quota) in enumerate(
            zip(names, strata, quotas)
        ):
            take = min(int(quota), int(len(pool)))
            if take <= 0:
                continue

            if take == len(pool):
                chosen = pool.copy()
            else:
                chosen = self.rng.choice(pool, take, replace=False)

            chosen = np.asarray(chosen, dtype=np.int64)
            selected_parts.append(chosen)
            selected_label_parts.append(
                np.full(len(chosen), label_id, dtype=np.uint8)
            )
            sampled_counts[name] += int(len(chosen))
            selected_set.update(chosen.tolist())

        current = sum(len(p) for p in selected_parts)
        deficit = self.num_point - current

        # Second pass: redistribute any missing quota. Priority intentionally
        # stays target-centric: remaining CORE -> CONTEXT -> BACKGROUND.
        if deficit > 0:
            for label_id, (name, pool) in enumerate(zip(names, strata)):
                if deficit <= 0:
                    break

                if selected_set:
                    selected_arr = np.fromiter(
                        selected_set,
                        dtype=np.int64,
                        count=len(selected_set),
                    )
                    remain = np.setdiff1d(
                        pool,
                        selected_arr,
                        assume_unique=False,
                    )
                else:
                    remain = pool

                take = min(deficit, len(remain))
                if take <= 0:
                    continue

                if take == len(remain):
                    chosen = remain.copy()
                else:
                    chosen = self.rng.choice(remain, take, replace=False)

                chosen = np.asarray(chosen, dtype=np.int64)
                selected_parts.append(chosen)
                selected_label_parts.append(
                    np.full(len(chosen), label_id, dtype=np.uint8)
                )
                sampled_counts[name] += int(len(chosen))
                selected_set.update(chosen.tolist())
                deficit -= int(len(chosen))

        if selected_parts:
            idx = np.concatenate(selected_parts, axis=0)
            labels = np.concatenate(selected_label_parts, axis=0)
        else:
            idx = np.zeros(0, dtype=np.int64)
            labels = np.zeros(0, dtype=np.uint8)

        replacement_points = 0

        # Only if the entire dense scene itself has fewer points than GraspNet
        # requires do we use replacement. V7/V8 dense-scene acquisition should
        # normally make this branch unnecessary.
        if len(idx) < self.num_point:
            missing = self.num_point - len(idx)

            if len(idx) == 0:
                source_idx = np.arange(n, dtype=np.int64)
                source_labels = np.full(n, 2, dtype=np.uint8)
            else:
                source_idx = idx
                source_labels = labels

            extra_pos = self.rng.choice(
                len(source_idx),
                missing,
                replace=True,
            )
            idx = np.concatenate([idx, source_idx[extra_pos]], axis=0)
            labels = np.concatenate(
                [labels, source_labels[extra_pos]],
                axis=0,
            )
            replacement_points = int(missing)

        # Randomize ordering before feeding PointNet++ so strata are not grouped.
        order = self.rng.permutation(len(idx))
        idx = idx[order]
        labels = labels[order]

        sampled = np.ascontiguousarray(pts[idx], dtype=np.float32)

        # Diagnostics: how much of the actual target surface made it into the
        # 20k input? object_support_radius is deliberately much tighter than the
        # context radius and is only a diagnostic, not another filter.
        support_radius = float(
            scfg.get("object_support_radius_m", 0.008)
        )
        sampled_dist, _ = tree.query(
            sampled.astype(np.float64),
            k=1,
            workers=-1,
        )
        support_count = int(np.count_nonzero(sampled_dist <= support_radius))

        # Measure what uniform sampling would have expected, useful when tuning
        # small-object bias without guessing.
        dense_support_count = int(
            np.count_nonzero(nearest_dist <= support_radius)
        )
        uniform_expected_support = (
            self.num_point * dense_support_count / max(n, 1)
        )

        stats = {
            "mode": "target_stratified",
            "dense_scene_points": int(n),
            "requested_points": int(self.num_point),
            "core_radius_m": float(core_radius),
            "context_radius_m": float(context_radius),
            "object_support_radius_m": float(support_radius),
            "requested_fraction": {
                "core": float(fractions[0]),
                "context": float(fractions[1]),
                "background": float(fractions[2]),
            },
            "requested_quota": {
                "core": int(quotas[0]),
                "context": int(quotas[1]),
                "background": int(quotas[2]),
            },
            "strata_available": {
                "core": int(len(core_idx)),
                "context": int(len(context_idx)),
                "background": int(len(background_idx)),
            },
            "strata_sampled": {
                "core": int(np.count_nonzero(labels == 0)),
                "context": int(np.count_nonzero(labels == 1)),
                "background": int(np.count_nonzero(labels == 2)),
            },
            "strata_sampled_fraction": {
                "core": float(np.mean(labels == 0)),
                "context": float(np.mean(labels == 1)),
                "background": float(np.mean(labels == 2)),
            },
            "sampled_object_support_points": int(support_count),
            "sampled_object_support_fraction": float(
                support_count / max(len(sampled), 1)
            ),
            "dense_object_support_points": int(dense_support_count),
            "uniform_expected_object_support_points": float(
                uniform_expected_support
            ),
            "object_support_gain_vs_uniform": float(
                support_count / max(uniform_expected_support, 1e-9)
            ),
            "sample_nearest_target_distance_m": {
                "p50": float(np.quantile(sampled_dist, 0.50)),
                "p90": float(np.quantile(sampled_dist, 0.90)),
                "p99": float(np.quantile(sampled_dist, 0.99)),
            },
            "unique_sampled_points": int(len(np.unique(idx))),
            "replacement_points": int(replacement_points),
        }

        return (
            sampled,
            np.ascontiguousarray(idx, dtype=np.int64),
            np.ascontiguousarray(labels, dtype=np.uint8),
            stats,
        )

    def _sample_scene(self, dense_scene, object_camera):
        scfg = self.cfg.get("sampling", {})
        mode = str(
            scfg.get("mode", "target_stratified")
        ).strip().lower()

        if mode == "uniform":
            return self._sample_uniform(dense_scene)

        if mode in (
            "target_stratified",
            "target-aware",
            "target_aware",
        ):
            return self._sample_target_stratified(
                dense_scene,
                object_camera,
            )

        raise ValueError(
            "Unsupported sampling.mode="
            f"{mode!r}; expected 'uniform' or 'target_stratified'"
        )

    def _infer(self, sampled_scene):
        tensor = torch.from_numpy(
            sampled_scene[np.newaxis, ...]
        ).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        t0 = time.perf_counter()

        with torch.no_grad():
            end_points = self.net({
                "point_clouds": tensor
            })
            preds = self.pred_decode(
                end_points
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        arr = (
            preds[0]
            .detach()
            .cpu()
            .numpy()
        )

        return (
            self.GraspGroup(arr),
            elapsed_ms,
        )

    def _width_filter(self, gg):
        if len(gg) == 0:
            return gg

        fcfg = self.cfg["filters"]

        wmin = float(
            fcfg.get(
                "gripper_width_min_m",
                0.005,
            )
        )
        wmax = float(
            fcfg.get(
                "gripper_width_max_m",
                0.085,
            )
        )

        keep = (
            (gg.widths >= wmin)
            & (gg.widths <= wmax)
        )

        return gg[keep]

    def _target_capture_filter(
        self,
        gg,
        object_camera,
    ):
        """
        Semantic gate only.

        We do NOT re-score grasps here.

        Conditions:
          1) grasp reference position is near SAM2 target cloud;
          2) at least N target points lie in the gripper closing volume.
        """
        if len(gg) == 0:
            return (
                gg,
                np.zeros(0),
                np.zeros(
                    0,
                    dtype=np.int32,
                ),
            )

        fcfg = self.cfg["filters"]

        max_center_dist = float(
            fcfg.get(
                "target_center_max_distance_m",
                0.075,
            )
        )

        min_inner = int(
            fcfg.get(
                "target_inner_min_points",
                3,
            )
        )

        tree = cKDTree(
            object_camera.astype(
                np.float64
            )
        )

        center_dist, _ = tree.query(
            gg.translations.astype(
                np.float64
            ),
            k=1,
            workers=-1,
        )

        target = object_camera.astype(
            np.float64
        )

        keep = np.zeros(
            len(gg),
            dtype=bool,
        )
        inner_counts = np.zeros(
            len(gg),
            dtype=np.int32,
        )

        # Matches the geometry used by GraspNet's model-free collision code.
        finger_length = float(
            fcfg.get(
                "target_capture_finger_length_m",
                0.06,
            )
        )

        for i in range(len(gg)):
            g = gg[i]

            if center_dist[i] > max_center_dist:
                continue

            R = np.asarray(
                g.rotation_matrix,
                dtype=np.float64,
            )
            t = np.asarray(
                g.translation,
                dtype=np.float64,
            )

            # Row-vector conversion into grasp coordinates.
            p = (
                target
                - t.reshape(1, 3)
            ) @ R

            width = float(g.width)
            height = max(
                float(g.height),
                0.02,
            )
            depth = float(g.depth)

            inside = (
                (p[:, 0] > depth - finger_length)
                & (p[:, 0] < depth)
                & (np.abs(p[:, 1]) < width / 2.0)
                & (np.abs(p[:, 2]) < height / 2.0)
            )

            count = int(
                np.count_nonzero(
                    inside
                )
            )
            inner_counts[i] = count

            if count >= min_inner:
                keep[i] = True

        return (
            gg[keep],
            center_dist[keep],
            inner_counts[keep],
        )

    def _collision_filter(
        self,
        gg,
        collision_scene_camera,
    ):
        if len(gg) == 0:
            return gg

        fcfg = self.cfg["filters"]

        detector = self.ModelFreeCollisionDetector(
            collision_scene_camera,
            voxel_size=float(
                fcfg.get(
                    "collision_voxel_size_m",
                    0.01,
                )
            ),
        )

        collision_mask = detector.detect(
            gg,
            approach_dist=float(
                fcfg.get(
                    "collision_approach_dist_m",
                    0.05,
                )
            ),
            collision_thresh=float(
                fcfg.get(
                    "collision_thresh",
                    0.01,
                )
            ),
        )

        return gg[
            ~collision_mask
        ]

    def _nms(self, gg):
        if len(gg) == 0:
            return gg

        fcfg = self.cfg["filters"]

        return gg.nms(
            translation_thresh=float(
                fcfg.get(
                    "nms_translation_thresh_m",
                    0.03,
                )
            ),
            rotation_thresh=math.radians(
                float(
                    fcfg.get(
                        "nms_rotation_thresh_deg",
                        30.0,
                    )
                )
            ),
        )

    @staticmethod
    def _sort_score(gg):
        if len(gg) == 0:
            return gg

        order = np.argsort(
            gg.scores.astype(
                np.float64
            )
        )[::-1]

        return gg[order]

    @staticmethod
    def _summary_scores(gg):
        if len(gg) == 0:
            return {
                "count": 0,
                "max": None,
                "median": None,
                "min": None,
            }

        s = gg.scores.astype(
            np.float64
        )

        return {
            "count": int(len(gg)),
            "max": float(np.max(s)),
            "median": float(np.median(s)),
            "min": float(np.min(s)),
        }

    @staticmethod
    def _summary_widths(gg):
        if len(gg) == 0:
            return {
                "count": 0,
                "min_m": None,
                "median_m": None,
                "max_m": None,
                "within_robot_limit": 0,
            }

        w = gg.widths.astype(
            np.float64
        )

        return {
            "count": int(len(w)),
            "min_m": float(np.min(w)),
            "median_m": float(np.median(w)),
            "max_m": float(np.max(w)),
        }

    @staticmethod
    def _save_gg(path, gg):
        np.save(
            path,
            gg.grasp_group_array,
        )

    def predict_npz(
        self,
        npz_path: str,
        save_dir: Optional[str] = None,
    ):
        total_t0 = time.perf_counter()

        npz_path = Path(
            npz_path
        ).resolve()

        data_npz = np.load(
            str(npz_path),
            allow_pickle=False,
        )

        data = {
            k: data_npz[k]
            for k in data_npz.files
        }

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
            k
            for k in required
            if k not in data
        ]

        if missing:
            raise RuntimeError(
                "NPZ uses the old sparse-scene contract. "
                f"Missing: {missing}. "
                "Regenerate this object with the modified "
                "wrist_rgbd_localizer before running GraspNet V7."
            )

        object_camera, object_keep = (
            self._sanitize_points(
                data["points_camera"],
                positive_z=True,
            )
        )

        object_base_raw = np.asarray(
            data["points_base"],
            dtype=np.float32,
        ).reshape(-1, 3)
        object_base = object_base_raw[
            object_keep
        ]

        dense_scene_camera, dense_keep = (
            self._sanitize_points(
                data[
                    "dense_scene_points_camera"
                ],
                positive_z=True,
            )
        )

        dense_scene_base_raw = np.asarray(
            data[
                "dense_scene_points_base"
            ],
            dtype=np.float32,
        ).reshape(-1, 3)
        dense_scene_base = (
            dense_scene_base_raw[
                dense_keep
            ]
        )

        collision_scene_camera, collision_keep = (
            self._sanitize_points(
                data["scene_points_camera"],
                positive_z=True,
            )
        )

        collision_scene_base_raw = np.asarray(
            data["scene_points_base"],
            dtype=np.float32,
        ).reshape(-1, 3)
        collision_scene_base = (
            collision_scene_base_raw[
                collision_keep
            ]
        )

        T_base_camera = np.asarray(
            data["T_base_camera"],
            dtype=np.float64,
        ).reshape(4, 4)

        table_z = float(
            np.asarray(
                data["table_z"]
            ).item()
        )

        icfg = self.cfg["input"]

        if len(object_camera) < int(
            icfg.get(
                "min_object_points",
                150,
            )
        ):
            raise RuntimeError(
                "Object cloud too small: "
                f"{len(object_camera)}"
            )

        if len(dense_scene_camera) < int(
            icfg.get(
                "min_dense_scene_points",
                12000,
            )
        ):
            raise RuntimeError(
                "Dense GraspNet scene too small: "
                f"{len(dense_scene_camera)} < "
                f"{icfg.get('min_dense_scene_points', 12000)}"
            )

        if len(collision_scene_camera) < int(
            icfg.get(
                "min_collision_scene_points",
                800,
            )
        ):
            raise RuntimeError(
                "Collision scene too small: "
                f"{len(collision_scene_camera)}"
            )

        (
            sampled_scene,
            sampled_scene_indices,
            sampled_scene_strata,
            sampling_stats,
        ) = self._sample_scene(
            dense_scene_camera,
            object_camera,
        )

        # 1) GraspNet on DENSE local RGB-D scene.
        gg_raw, inference_ms = self._infer(
            sampled_scene
        )

        # 2) Actual Robotiq opening capability.
        gg_width = self._width_filter(
            gg_raw
        )

        # 3) Keep grasps that capture the requested SAM2 target.
        (
            gg_target,
            _,
            _,
        ) = self._target_capture_filter(
            gg_width,
            object_camera,
        )

        # 4) Collision against the SPARSE local scene.
        gg_collision = (
            self._collision_filter(
                gg_target,
                collision_scene_camera,
            )
        )

        # 5) NMS after collision.
        gg_nms = self._nms(
            gg_collision
        )

        # 6) Original GraspNet score only.
        gg_final = self._sort_score(
            gg_nms
        )

        top_n = int(
            self.cfg["filters"].get(
                "top_n",
                50,
            )
        )
        if len(gg_final) > top_n:
            gg_final = gg_final[
                :top_n
            ]

        label = (
            str(data["label"].item())
            if "label" in data
            else ""
        )

        R_bc = T_base_camera[:3, :3]
        t_bc = T_base_camera[:3, 3]

        if len(gg_final) > 0:
            final_tree = cKDTree(
                object_camera.astype(
                    np.float64
                )
            )
            final_dist, _ = (
                final_tree.query(
                    gg_final.translations.astype(
                        np.float64
                    ),
                    k=1,
                    workers=-1,
                )
            )
        else:
            final_dist = np.zeros(
                0,
                dtype=np.float64,
            )

        grasps = []

        for i in range(
            len(gg_final)
        ):
            g = gg_final[i]

            R_cg = np.asarray(
                g.rotation_matrix,
                dtype=np.float64,
            )
            t_cg = np.asarray(
                g.translation,
                dtype=np.float64,
            )

            R_bg = R_bc @ R_cg
            t_bg = R_bc @ t_cg + t_bc

            approach = R_bg[:, 0]
            down = np.array(
                [0.0, 0.0, -1.0],
                dtype=np.float64,
            )

            approach_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            approach @ down,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )

            grasps.append({
                "rank": int(i),
                "model_score": float(
                    g.score
                ),
                "width_m": float(
                    g.width
                ),
                "height_m": float(
                    g.height
                ),
                "depth_m": float(
                    g.depth
                ),
                "translation_camera_m":
                    t_cg.tolist(),
                "rotation_camera_grasp":
                    R_cg.tolist(),
                "translation_base_grasp_m":
                    t_bg.tolist(),
                "rotation_base_grasp":
                    R_bg.tolist(),
                "approach_axis_base":
                    approach.tolist(),
                "jaw_axis_base":
                    R_bg[:, 1].tolist(),
                "approach_angle_from_down_deg":
                    approach_deg,
                "target_nearest_distance_m":
                    float(
                        final_dist[i]
                    ),
            })

        total_ms = (
            time.perf_counter()
            - total_t0
        ) * 1000.0

        wmin = float(
            self.cfg["filters"].get(
                "gripper_width_min_m",
                0.005,
            )
        )
        wmax = float(
            self.cfg["filters"].get(
                "gripper_width_max_m",
                0.085,
            )
        )

        raw_widths = (
            gg_raw.widths.astype(
                np.float64
            )
            if len(gg_raw)
            else np.zeros(
                0,
                dtype=np.float64,
            )
        )

        summary = {
            "status": "ok",
            "version":
                "v8_target_aware_stratified_sampling",
            "label": label,
            "proposal_source":
                "dense_scene_points_camera_target_stratified",
            "collision_source":
                "scene_points_camera",
            "ranking_policy":
                "original_graspnet_model_score_descending",

            "sampling": sampling_stats,

            "counts": {
                "object_points":
                    int(
                        len(object_camera)
                    ),
                "dense_scene_points":
                    int(
                        len(
                            dense_scene_camera
                        )
                    ),
                "collision_scene_points":
                    int(
                        len(
                            collision_scene_camera
                        )
                    ),
                "sampled_scene_points":
                    int(
                        len(sampled_scene)
                    ),
                "raw_grasps":
                    int(
                        len(gg_raw)
                    ),
                "after_width":
                    int(
                        len(gg_width)
                    ),
                "after_target_capture":
                    int(
                        len(gg_target)
                    ),
                "after_scene_collision":
                    int(
                        len(gg_collision)
                    ),
                "after_nms":
                    int(
                        len(gg_nms)
                    ),
                "final":
                    int(
                        len(gg_final)
                    ),
            },

            "raw_width_diagnostics": {
                "min_m":
                    float(
                        np.min(raw_widths)
                    )
                    if len(raw_widths)
                    else None,
                "median_m":
                    float(
                        np.median(raw_widths)
                    )
                    if len(raw_widths)
                    else None,
                "max_m":
                    float(
                        np.max(raw_widths)
                    )
                    if len(raw_widths)
                    else None,
                "within_robot_limit":
                    int(
                        np.count_nonzero(
                            (
                                raw_widths
                                >= wmin
                            )
                            &
                            (
                                raw_widths
                                <= wmax
                            )
                        )
                    ),
            },

            "score_summary": {
                "raw":
                    self._summary_scores(
                        gg_raw
                    ),
                "after_width":
                    self._summary_scores(
                        gg_width
                    ),
                "after_target_capture":
                    self._summary_scores(
                        gg_target
                    ),
                "after_scene_collision":
                    self._summary_scores(
                        gg_collision
                    ),
                "after_nms":
                    self._summary_scores(
                        gg_nms
                    ),
                "final":
                    self._summary_scores(
                        gg_final
                    ),
            },

            "latency_ms": {
                "network_inference":
                    float(
                        inference_ms
                    ),
                "total":
                    float(
                        total_ms
                    ),
            },

            "filters":
                self.cfg["filters"],

            "table_z_m":
                table_z,

            "grasps":
                grasps,
        }

        if save_dir is not None:
            out = Path(
                save_dir
            ).resolve()

            out.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._save_gg(
                out
                / "grasps_raw_camera.npy",
                gg_raw,
            )
            self._save_gg(
                out
                / "grasps_width_camera.npy",
                gg_width,
            )
            self._save_gg(
                out
                / "grasps_target_camera.npy",
                gg_target,
            )
            self._save_gg(
                out
                / "grasps_collision_free_camera.npy",
                gg_collision,
            )
            self._save_gg(
                out
                / "grasps_nms_camera.npy",
                gg_nms,
            )
            self._save_gg(
                out
                / "grasps_final_camera.npy",
                gg_final,
            )
            self._save_gg(
                out
                / "grasps_camera.npy",
                gg_final,
            )

            with open(
                out / "result.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    summary,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # Sampling debug: this is the exact 20k network input plus the
            # stratum label of every sampled point:
            #   0=core, 1=context, 2=background.
            np.savez_compressed(
                out / "sampling_debug.npz",
                sampled_scene_camera=sampled_scene,
                sampled_scene_indices=sampled_scene_indices,
                sampled_scene_strata=sampled_scene_strata,
                object_points_camera=object_camera,
                T_base_camera=T_base_camera,
            )

            # Keep grasp visualization lightweight:
            # gray collision scene + green target.
            np.savez_compressed(
                out / "grasp_debug_scene.npz",
                scene_points_base=
                    collision_scene_base,
                object_points_base=
                    object_base,
                scene_points_camera=
                    collision_scene_camera,
                object_points_camera=
                    object_camera,
                T_base_camera=
                    T_base_camera,
                table_z=np.array(
                    table_z,
                    dtype=np.float64,
                ),
            )

        return gg_final, summary
