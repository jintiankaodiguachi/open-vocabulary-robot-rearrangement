#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
import pycocotools.mask as mask_util

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, String, Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from tf2_ros import Buffer, TransformListener

from open_vocab_interfaces.msg import SceneObject


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.dot(q, q))
    if n < 1e-14:
        raise ValueError("zero quaternion")
    q /= math.sqrt(n)
    x, y, z, w = q

    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x*x + y*y)],
    ], dtype=np.float64)


def decode_coco_rle(rle: Dict[str, Any]) -> np.ndarray:
    rr = {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"],
    }
    if isinstance(rr["counts"], str):
        rr["counts"] = rr["counts"].encode("utf-8")

    mask = mask_util.decode(rr)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or len(xyz) == 0:
        return xyz, rgb

    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)

    counts = np.bincount(inverse).astype(np.float64)
    n = len(counts)

    xyz_out = np.empty((n, 3), dtype=np.float64)
    rgb_out = np.empty((n, 3), dtype=np.float64)

    for i in range(3):
        xyz_out[:, i] = np.bincount(
            inverse,
            weights=xyz[:, i],
            minlength=n,
        ) / counts

        rgb_out[:, i] = np.bincount(
            inverse,
            weights=rgb[:, i],
            minlength=n,
        ) / counts

    return xyz_out.astype(np.float32), np.clip(rgb_out, 0, 255).astype(np.uint8)


class WristRGBDLocalizer(Node):
    """
    Stage 2B:
        wrist RGB + aligned depth
          -> A6000 Grounded-SAM2
          -> precise local mask
          -> masked depth
          -> object point cloud
          -> camera optical XYZ
          -> base_link XYZ

    The node deliberately does NOT generate a grasp. Its output is the precise
    object cloud that becomes the input to GraspNet in Stage 2C.
    """

    def __init__(self):
        super().__init__("open_vocab_wrist_rgbd_localizer")

        self.cb_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # Topics / frames.
        self.color_topic = self.declare_parameter(
            "color_topic",
            "/camera/camera/color/image_raw",
        ).value
        self.depth_topic = self.declare_parameter(
            "depth_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
        ).value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic",
            "/camera/camera/color/camera_info",
        ).value
        self.scene_topic = self.declare_parameter(
            "scene_topic",
            "/open_vocab/scene_object",
        ).value

        self.base_frame = self.declare_parameter(
            "base_frame",
            "base_link",
        ).value
        self.camera_frame = self.declare_parameter(
            "camera_frame",
            "camera_color_optical_frame",
        ).value

        # A6000.
        self.a6000_url = self.declare_parameter(
            "a6000_segment_url",
            "http://ai-server.local:8090/api/v1/segment_uploaded",
        ).value
        self.http_connect_timeout_s = float(self.declare_parameter(
            "http_connect_timeout_s", 2.0
        ).value)
        self.http_read_timeout_s = float(self.declare_parameter(
            "http_read_timeout_s", 30.0
        ).value)
        self.jpeg_quality = int(self.declare_parameter(
            "jpeg_quality", 92
        ).value)

        # Sync / freshness.
        self.sync_queue_size = int(self.declare_parameter(
            "sync_queue_size", 10
        ).value)
        self.sync_slop_s = float(self.declare_parameter(
            "sync_slop_s", 0.08
        ).value)
        self.max_frame_age_s = float(self.declare_parameter(
            "max_frame_age_s", 1.0
        ).value)

        # Local target selection.
        self.local_min_confidence = float(self.declare_parameter(
            "local_min_confidence", 0.35
        ).value)
        self.center_distance_weight = float(self.declare_parameter(
            "center_distance_weight", 0.15
        ).value)
        self.max_center_distance_norm = float(self.declare_parameter(
            "max_center_distance_norm", 0.85
        ).value)

        # Depth / mask.
        self.mask_erode_px = int(self.declare_parameter(
            "mask_erode_px", 2
        ).value)
        self.depth_min_m = float(self.declare_parameter(
            "depth_min_m", 0.15
        ).value)
        self.depth_max_m = float(self.declare_parameter(
            "depth_max_m", 1.50
        ).value)
        self.depth_percentile_low = float(self.declare_parameter(
            "depth_percentile_low", 1.0
        ).value)
        self.depth_percentile_high = float(self.declare_parameter(
            "depth_percentile_high", 99.0
        ).value)
        self.pixel_stride = max(1, int(self.declare_parameter(
            "pixel_stride", 1
        ).value))
        self.voxel_size_m = float(self.declare_parameter(
            "voxel_size_m", 0.003
        ).value)
        self.min_cloud_points = int(self.declare_parameter(
            "min_cloud_points", 200
        ).value)

        # Stage 2B -> Stage 2C point-cloud contract.
        #
        # 1) object cloud:
        #       SAM2 target only. Used for semantic association.
        #
        # 2) dense scene cloud:
        #       dense RGB-D points around the target, NO voxel downsampling.
        #       This is the GraspNet NETWORK INPUT.
        #
        # 3) collision scene cloud:
        #       voxel-downsampled copy of the same local scene.
        #       This is used by ModelFreeCollisionDetector / RViz debug.
        #
        # Keep old scene_points_* names for the collision cloud so existing
        # downstream tools remain backward-compatible.
        self.table_z = float(self.declare_parameter(
            "table_z", -0.031
        ).value)

        self.scene_bbox_expand_px = int(self.declare_parameter(
            "scene_bbox_expand_px", 150
        ).value)

        # Dense scene: stride=1 means every valid aligned-depth pixel in ROI.
        self.dense_scene_pixel_stride = max(1, int(self.declare_parameter(
            "dense_scene_pixel_stride", 1
        ).value))
        self.dense_scene_min_points = int(self.declare_parameter(
            "dense_scene_min_points", 12000
        ).value)
        self.dense_scene_max_points = int(self.declare_parameter(
            "dense_scene_max_points", 120000
        ).value)

        # Depth window around the target. It must include the tabletop/support
        # surface and nearby obstacles, not only the SAM2 object surface.
        self.scene_depth_margin_near_m = float(self.declare_parameter(
            "scene_depth_margin_near_m", 0.20
        ).value)
        self.scene_depth_margin_far_m = float(self.declare_parameter(
            "scene_depth_margin_far_m", 0.20
        ).value)

        # Collision/debug scene is intentionally sparse.
        self.scene_voxel_size_m = float(self.declare_parameter(
            "scene_voxel_size_m", 0.005
        ).value)
        self.scene_min_points = int(self.declare_parameter(
            "scene_min_points", 800
        ).value)

        # Base-frame Z crop rejects points far below the physical table or far
        # above the tabletop workspace.
        self.scene_base_z_min_offset_m = float(self.declare_parameter(
            "scene_base_z_min_offset_m", -0.025
        ).value)
        self.scene_base_z_max_offset_m = float(self.declare_parameter(
            "scene_base_z_max_offset_m", 0.35
        ).value)

        # TF.
        self.tf_timeout_s = float(self.declare_parameter(
            "tf_timeout_s", 0.5
        ).value)
        self.use_latest_tf_fallback = bool(self.declare_parameter(
            "use_latest_tf_fallback", True
        ).value)

        # Output.
        self.output_dir = Path(self.declare_parameter(
            "output_dir",
            "/opt/open-vocabulary-robot-rearrangement/perception/local_outputs",
        ).value).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto trigger is OFF for Day 6 debugging.
        self.auto_trigger = bool(self.declare_parameter(
            "auto_trigger", False
        ).value)
        self.observation_done_topic = self.declare_parameter(
            "observation_done_topic",
            "/open_vocab/observation_motion_done",
        ).value

        self.session = requests.Session()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self.latest_color_msg: Optional[Image] = None
        self.latest_depth_msg: Optional[Image] = None
        self.latest_pair_receive_monotonic = 0.0
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_scene: Optional[SceneObject] = None

        # CameraInfo is typically latched-ish / slow-changing; cache it separately.
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.on_camera_info,
            10,
            callback_group=self.cb_group,
        )

        self.create_subscription(
            SceneObject,
            self.scene_topic,
            self.on_scene_object,
            10,
            callback_group=self.cb_group,
        )

        color_sub = Subscriber(
            self,
            Image,
            self.color_topic,
            callback_group=self.cb_group,
        )
        depth_sub = Subscriber(
            self,
            Image,
            self.depth_topic,
            callback_group=self.cb_group,
        )

        self.sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_s,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.on_rgbd_pair)

        self.trigger_srv = self.create_service(
            Trigger,
            "/open_vocab/local_perception/trigger",
            self.on_trigger,
            callback_group=self.cb_group,
        )

        if self.auto_trigger:
            self.create_subscription(
                Bool,
                self.observation_done_topic,
                self.on_observation_done,
                10,
                callback_group=self.cb_group,
            )

        self.cloud_camera_pub = self.create_publisher(
            PointCloud2,
            "/open_vocab/local/object_cloud_camera",
            10,
        )
        self.cloud_base_pub = self.create_publisher(
            PointCloud2,
            "/open_vocab/local/object_cloud_base",
            10,
        )
        self.scene_cloud_camera_pub = self.create_publisher(
            PointCloud2,
            "/open_vocab/local/scene_cloud_camera",
            10,
        )
        self.scene_cloud_base_pub = self.create_publisher(
            PointCloud2,
            "/open_vocab/local/scene_cloud_base",
            10,
        )
        self.dense_scene_cloud_camera_pub = self.create_publisher(
            PointCloud2,
            "/open_vocab/local/dense_scene_cloud_camera",
            10,
        )
        self.dense_scene_cloud_base_pub = self.create_publisher(
            PointCloud2,
            "/open_vocab/local/dense_scene_cloud_base",
            10,
        )
        self.centroid_pub = self.create_publisher(
            PoseStamped,
            "/open_vocab/local/centroid_base",
            10,
        )
        self.mask_pub = self.create_publisher(
            Image,
            "/open_vocab/local/mask",
            10,
        )
        self.debug_pub = self.create_publisher(
            Image,
            "/open_vocab/local/debug_image",
            10,
        )
        self.marker_pub = self.create_publisher(
            Marker,
            "/open_vocab/local/centroid_marker",
            10,
        )
        self.result_pub = self.create_publisher(
            String,
            "/open_vocab/local/result_json",
            10,
        )

        self.get_logger().info(
            "WristRGBDLocalizer ready: "
            f"color={self.color_topic} "
            f"depth={self.depth_topic} "
            f"camera_info={self.camera_info_topic} "
            f"A6000={self.a6000_url}"
        )
        self.get_logger().warn(
            f"auto_trigger={self.auto_trigger}. "
            "For the first Day-6 test, manual Trigger service is recommended."
        )

    def on_camera_info(self, msg: CameraInfo):
        with self.lock:
            self.latest_camera_info = msg

    def on_scene_object(self, msg: SceneObject):
        with self.lock:
            self.latest_scene = msg

    def on_rgbd_pair(self, color_msg: Image, depth_msg: Image):
        with self.lock:
            self.latest_color_msg = color_msg
            self.latest_depth_msg = depth_msg
            self.latest_pair_receive_monotonic = time.monotonic()

    def on_observation_done(self, msg: Bool):
        if not msg.data:
            return

        self.get_logger().info(
            "Observation motion completion received; starting local RGB-D perception."
        )

        # MultiThreadedExecutor + Reentrant callback group allows this callback
        # to perform the HTTP inference while image callbacks keep running.
        try:
            ok, message = self.process_once()
        except Exception as exc:
            # Defense in depth: even an unexpected bug must not kill the ROS node.
            message = f"{type(exc).__name__}: {exc}"
            ok = False
            self.get_logger().error(
                "Unhandled exception in auto-trigger callback:\n"
                + traceback.format_exc()
            )

        if ok:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)

    def on_trigger(self, request, response):
        del request

        try:
            ok, message = self.process_once()
        except Exception as exc:
            # Never let a service callback exception terminate the executor.
            ok = False
            message = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(
                "Unhandled exception in Trigger callback:\n"
                + traceback.format_exc()
            )

        response.success = ok
        response.message = message
        return response

    def snapshot_inputs(self):
        with self.lock:
            color_msg = self.latest_color_msg
            depth_msg = self.latest_depth_msg
            info = self.latest_camera_info
            scene = self.latest_scene
            age = (
                time.monotonic() - self.latest_pair_receive_monotonic
                if self.latest_pair_receive_monotonic > 0.0
                else float("inf")
            )

        return color_msg, depth_msg, info, scene, age

    def request_segmentation(self, image_bgr: np.ndarray, label: str) -> Dict[str, Any]:
        ok, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")

        response = self.session.post(
            self.a6000_url,
            files={
                "image": (
                    "wrist_rgb.jpg",
                    encoded.tobytes(),
                    "image/jpeg",
                )
            },
            data={
                "text_prompt": label,
                "include_mask_rle": "true",
                "save_debug_image": "false",
            },
            timeout=(
                self.http_connect_timeout_s,
                self.http_read_timeout_s,
            ),
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"A6000 returned HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        data = response.json()
        if "detections" not in data:
            raise RuntimeError(
                f"A6000 response has no detections field: {data}"
            )
        return data

    def select_local_detection(
        self,
        detections,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        cx_img = 0.5 * width
        cy_img = 0.5 * height
        diag_half = max(
            1.0,
            math.hypot(cx_img, cy_img),
        )

        candidates = []

        for det in detections:
            conf = float(det.get("confidence", 0.0))
            if conf < self.local_min_confidence:
                continue
            if "mask_rle" not in det:
                continue

            c = det.get("centroid_uv")
            if c is None or len(c) != 2:
                continue

            u = float(c[0])
            v = float(c[1])

            center_dist_norm = math.hypot(
                u - cx_img,
                v - cy_img,
            ) / diag_half

            if center_dist_norm > self.max_center_distance_norm:
                continue

            score = (
                conf
                - self.center_distance_weight * center_dist_norm
            )

            candidates.append(
                (score, conf, -center_dist_norm, det)
            )

        if not candidates:
            raise RuntimeError(
                "No local detection survived confidence/center/mask filtering"
            )

        candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
        return candidates[0][3]

    def depth_to_meters(self, depth_msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough",
        )

        enc = str(depth_msg.encoding).upper()

        if enc in ("16UC1", "MONO16"):
            return depth.astype(np.float32) * 0.001

        if enc == "32FC1":
            return depth.astype(np.float32)

        raise RuntimeError(
            f"Unsupported depth encoding '{depth_msg.encoding}'. "
            "Expected aligned depth as 16UC1(mm) or 32FC1(m)."
        )

    def get_transform_matrix(
        self,
        color_msg: Image,
    ) -> np.ndarray:
        source_frame = self.camera_frame

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time.from_msg(color_msg.header.stamp),
                timeout=Duration(seconds=self.tf_timeout_s),
            )
        except Exception as first_exc:
            if not self.use_latest_tf_fallback:
                raise RuntimeError(
                    f"TF {self.base_frame}<-{source_frame} at image stamp failed: "
                    f"{first_exc}"
                ) from first_exc

            self.get_logger().warn(
                "Timestamped camera TF unavailable; falling back to latest TF. "
                f"Reason: {first_exc}"
            )

            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_s),
            )

        t = tf.transform.translation
        q = tf.transform.rotation

        R = quaternion_to_rotation_matrix(
            q.x, q.y, q.z, q.w
        )

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def build_object_cloud(
        self,
        image_bgr: np.ndarray,
        depth_m: np.ndarray,
        camera_info: CameraInfo,
        mask: np.ndarray,
    ):
        h, w = image_bgr.shape[:2]

        if depth_m.shape[:2] != (h, w):
            raise RuntimeError(
                f"Aligned depth size {depth_m.shape[1]}x{depth_m.shape[0]} "
                f"does not match color {w}x{h}. "
                "Use aligned_depth_to_color, not raw depth."
            )

        if mask.shape != (h, w):
            raise RuntimeError(
                f"SAM mask size {mask.shape[::-1]} "
                f"does not match color {w}x{h}"
            )

        raw_mask = mask.astype(bool)
        raw_mask_pixels = int(np.count_nonzero(raw_mask))

        if raw_mask_pixels == 0:
            raise RuntimeError("SAM mask is empty")

        work_mask = raw_mask.astype(np.uint8)

        if self.mask_erode_px > 0:
            k = 2 * self.mask_erode_px + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            eroded = cv2.erode(work_mask, kernel, iterations=1)

            # Do not destroy a small object mask.
            if int(eroded.sum()) >= max(50, int(0.30 * work_mask.sum())):
                work_mask = eroded

        processed_mask_pixels = int(np.count_nonzero(work_mask))

        ys, xs = np.nonzero(work_mask)

        if self.pixel_stride > 1 and len(xs) > 0:
            keep = np.arange(len(xs)) % self.pixel_stride == 0
            xs = xs[keep]
            ys = ys[keep]

        # This is the denominator for depth coverage. It is the number of
        # processed-mask pixels actually sampled before invalid-depth removal.
        mask_depth_samples = int(len(xs))

        z_all = depth_m[ys, xs]

        valid = (
            np.isfinite(z_all)
            & (z_all >= self.depth_min_m)
            & (z_all <= self.depth_max_m)
        )

        valid_depth_samples = int(np.count_nonzero(valid))
        valid_depth_ratio = (
            float(valid_depth_samples) / float(mask_depth_samples)
            if mask_depth_samples > 0
            else 0.0
        )

        xs = xs[valid]
        ys = ys[valid]
        z = z_all[valid]

        if len(z) < self.min_cloud_points:
            raise RuntimeError(
                "Too few valid masked depth points before filtering: "
                f"{len(z)}; mask_depth_samples={mask_depth_samples}; "
                f"valid_depth_ratio={valid_depth_ratio:.3f}"
            )

        # Remove only extreme depth outliers. We intentionally do not collapse
        # the cloud to one median-depth slab because object surfaces are 3-D.
        lo = float(np.percentile(z, self.depth_percentile_low))
        hi = float(np.percentile(z, self.depth_percentile_high))

        depth_keep = (z >= lo) & (z <= hi)

        xs = xs[depth_keep]
        ys = ys[depth_keep]
        z = z[depth_keep]

        post_percentile_depth_samples = int(len(z))

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])

        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(
                f"Invalid color CameraInfo intrinsics fx={fx}, fy={fy}"
            )

        x = (xs.astype(np.float32) - cx) * z / fx
        y = (ys.astype(np.float32) - cy) * z / fy

        xyz_camera = np.stack([x, y, z], axis=1).astype(np.float32)
        rgb = image_bgr[ys, xs][:, ::-1].copy()

        pre_voxel_points = int(len(xyz_camera))

        xyz_camera, rgb = voxel_downsample(
            xyz_camera,
            rgb,
            self.voxel_size_m,
        )

        post_voxel_points = int(len(xyz_camera))

        if len(xyz_camera) < self.min_cloud_points:
            raise RuntimeError(
                f"Too few object cloud points after voxelization: "
                f"{len(xyz_camera)} < {self.min_cloud_points}; "
                f"valid_depth_ratio={valid_depth_ratio:.3f}"
            )

        depth_stats = {
            "raw_mask_pixels": raw_mask_pixels,
            "processed_mask_pixels": processed_mask_pixels,
            "mask_depth_samples": mask_depth_samples,
            "valid_depth_samples": valid_depth_samples,
            "valid_depth_ratio": valid_depth_ratio,
            "post_percentile_depth_samples": post_percentile_depth_samples,
            "pre_voxel_points": pre_voxel_points,
            "post_voxel_points": post_voxel_points,
            "depth_percentile_low_m": lo,
            "depth_percentile_high_m": hi,
        }

        return (
            xyz_camera,
            rgb,
            work_mask.astype(bool),
            depth_stats,
        )

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise RuntimeError("cannot compute bbox from empty mask")
        return [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

    def build_scene_clouds(
        self,
        image_bgr: np.ndarray,
        depth_m: np.ndarray,
        camera_info: CameraInfo,
        target_mask: np.ndarray,
        target_bbox_xyxy,
        target_median_depth_m: float,
        T_base_camera: np.ndarray,
    ):
        """
        Build TWO scene clouds from the SAME aligned RGB-D frame.

        dense scene:
            Dense, target-centered local RGB-D scene. No voxel downsampling.
            This is the input for the pretrained GraspNet network.

        collision scene:
            Voxel-downsampled copy of the dense local scene.
            This is used for collision checking and lightweight visualization.

        Both clouds contain:
            - target object
            - tabletop/support surface
            - nearby obstacles / local context

        The SAM2 mask is used only to define the target and ROI. It is NOT
        applied as a mask to the scene cloud.
        """
        h, w = image_bgr.shape[:2]

        if depth_m.shape[:2] != (h, w):
            raise RuntimeError(
                f"Aligned depth size {depth_m.shape[1]}x{depth_m.shape[0]} "
                f"does not match color {w}x{h}"
            )

        if target_bbox_xyxy is None or len(target_bbox_xyxy) != 4:
            target_bbox_xyxy = self._bbox_from_mask(target_mask)

        x1, y1, x2, y2 = [
            int(round(float(v)))
            for v in target_bbox_xyxy
        ]

        margin = max(0, self.scene_bbox_expand_px)

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w - 1, x2 + margin)
        y2 = min(h - 1, y2 + margin)

        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(
                f"invalid dense scene ROI: {(x1, y1, x2, y2)}"
            )

        stride = self.dense_scene_pixel_stride

        ys_grid, xs_grid = np.mgrid[
            y1:y2 + 1:stride,
            x1:x2 + 1:stride,
        ]

        xs = xs_grid.reshape(-1)
        ys = ys_grid.reshape(-1)

        z_all = depth_m[ys, xs]

        near = max(
            self.depth_min_m,
            target_median_depth_m - self.scene_depth_margin_near_m,
        )
        far = min(
            self.depth_max_m,
            target_median_depth_m + self.scene_depth_margin_far_m,
        )

        valid = (
            np.isfinite(z_all)
            & (z_all >= near)
            & (z_all <= far)
        )

        xs = xs[valid]
        ys = ys[valid]
        z = z_all[valid]

        if len(z) == 0:
            raise RuntimeError(
                "Dense scene ROI contains no valid aligned depth"
            )

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])

        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(
                f"Invalid color CameraInfo intrinsics fx={fx}, fy={fy}"
            )

        x = (xs.astype(np.float32) - cx) * z / fx
        y = (ys.astype(np.float32) - cy) * z / fy

        dense_camera = np.stack(
            [x, y, z],
            axis=1,
        ).astype(np.float32)

        dense_rgb = image_bgr[ys, xs][:, ::-1].copy()

        dense_base = (
            dense_camera.astype(np.float64)
            @ T_base_camera[:3, :3].T
            + T_base_camera[:3, 3]
        ).astype(np.float32)

        # Physical tabletop workspace crop in base_link.
        z_min = self.table_z + self.scene_base_z_min_offset_m
        z_max = self.table_z + self.scene_base_z_max_offset_m

        keep = (
            np.isfinite(dense_camera).all(axis=1)
            & np.isfinite(dense_base).all(axis=1)
            & (dense_camera[:, 2] > 0.0)
            & (dense_base[:, 2] >= z_min)
            & (dense_base[:, 2] <= z_max)
        )

        dense_camera = dense_camera[keep]
        dense_base = dense_base[keep]
        dense_rgb = dense_rgb[keep]

        # A hard memory cap only. This is NOT voxel downsampling and is only
        # expected to activate for unusually large ROIs/resolutions.
        pre_cap_points = int(len(dense_camera))
        if (
            self.dense_scene_max_points > 0
            and len(dense_camera) > self.dense_scene_max_points
        ):
            idx = np.linspace(
                0,
                len(dense_camera) - 1,
                num=self.dense_scene_max_points,
                dtype=np.int64,
            )
            dense_camera = dense_camera[idx]
            dense_base = dense_base[idx]
            dense_rgb = dense_rgb[idx]

        if len(dense_camera) < self.dense_scene_min_points:
            raise RuntimeError(
                "Dense GraspNet scene too small: "
                f"{len(dense_camera)} < {self.dense_scene_min_points}; "
                f"ROI={(x1, y1, x2, y2)}, stride={stride}, "
                f"depth_window=({near:.3f},{far:.3f})m. "
                "Do not continue to GraspNet with a sparse repeated cloud."
            )

        # Collision cloud is derived from dense scene AFTER all geometry checks.
        collision_camera, collision_rgb = voxel_downsample(
            dense_camera,
            dense_rgb,
            self.scene_voxel_size_m,
        )

        collision_base = (
            collision_camera.astype(np.float64)
            @ T_base_camera[:3, :3].T
            + T_base_camera[:3, 3]
        ).astype(np.float32)

        if len(collision_camera) < self.scene_min_points:
            raise RuntimeError(
                "Collision scene cloud too small: "
                f"{len(collision_camera)} < {self.scene_min_points}; "
                f"dense={len(dense_camera)}, "
                f"voxel={self.scene_voxel_size_m:.4f}m"
            )

        stats = {
            "scene_roi_xyxy": [x1, y1, x2, y2],
            "scene_depth_near_m": float(near),
            "scene_depth_far_m": float(far),
            "dense_scene_pixel_stride": int(stride),
            "dense_scene_points_before_cap": pre_cap_points,
            "dense_scene_points": int(len(dense_camera)),
            "collision_scene_points": int(len(collision_camera)),
            "collision_scene_voxel_size_m": float(
                self.scene_voxel_size_m
            ),
        }

        return (
            dense_camera,
            dense_base,
            dense_rgb,
            collision_camera,
            collision_base,
            collision_rgb,
            stats,
        )

    def publish_debug(
        self,
        color_msg: Image,
        image_bgr: np.ndarray,
        det: Dict[str, Any],
        mask: np.ndarray,
    ):
        debug = image_bgr.copy()

        overlay = debug.copy()
        overlay[mask] = (
            0.45 * overlay[mask]
            + 0.55 * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)

        debug = cv2.addWeighted(
            overlay,
            0.55,
            debug,
            0.45,
            0.0,
        )

        bbox = det.get("bbox_xyxy")
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2,
            )

        c = det.get("centroid_uv")
        if c is not None:
            cv2.circle(
                debug,
                (int(round(float(c[0]))), int(round(float(c[1])))),
                5,
                (0, 0, 255),
                -1,
            )

        text = (
            f'{det.get("label", "")} '
            f'{float(det.get("confidence", 0.0)):.2f}'
        )
        cv2.putText(
            debug,
            text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        debug_msg = self.bridge.cv2_to_imgmsg(
            debug,
            encoding="bgr8",
        )
        debug_msg.header = color_msg.header
        self.debug_pub.publish(debug_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(
            (mask.astype(np.uint8) * 255),
            encoding="mono8",
        )
        mask_msg.header = color_msg.header
        self.mask_pub.publish(mask_msg)

    def process_once(self) -> Tuple[bool, str]:
        t_total = time.perf_counter()

        color_msg, depth_msg, camera_info, scene, frame_age = (
            self.snapshot_inputs()
        )

        if scene is None:
            return False, "No SceneObject received yet"

        if color_msg is None or depth_msg is None:
            return False, "No synchronized wrist RGB-D pair received yet"

        if camera_info is None:
            return False, "No color CameraInfo received yet"

        if frame_age > self.max_frame_age_s:
            return False, (
                f"Latest RGB-D pair is stale: age={frame_age:.3f}s "
                f"> {self.max_frame_age_s:.3f}s"
            )

        label = str(scene.label).strip()
        if not label:
            return False, "Latest SceneObject has an empty label"

        try:
            t_decode = time.perf_counter()

            image_bgr = self.bridge.imgmsg_to_cv2(
                color_msg,
                desired_encoding="bgr8",
            )
            depth_m = self.depth_to_meters(depth_msg)

            decode_ms = (time.perf_counter() - t_decode) * 1000.0

            h, w = image_bgr.shape[:2]

            if (
                int(camera_info.width) not in (0, w)
                or int(camera_info.height) not in (0, h)
            ):
                raise RuntimeError(
                    f"CameraInfo resolution "
                    f"{camera_info.width}x{camera_info.height} "
                    f"does not match RGB {w}x{h}"
                )

            t_ai = time.perf_counter()
            ai = self.request_segmentation(image_bgr, label)
            ai_roundtrip_ms = (time.perf_counter() - t_ai) * 1000.0

            det = self.select_local_detection(
                ai["detections"],
                width=w,
                height=h,
            )

            mask = decode_coco_rle(det["mask_rle"])

            t_cloud = time.perf_counter()
            (
                xyz_camera,
                rgb,
                processed_mask,
                depth_stats,
            ) = self.build_object_cloud(
                image_bgr=image_bgr,
                depth_m=depth_m,
                camera_info=camera_info,
                mask=mask,
            )

            T_base_camera = self.get_transform_matrix(color_msg)

            xyz_base = (
                xyz_camera.astype(np.float64)
                @ T_base_camera[:3, :3].T
                + T_base_camera[:3, 3]
            ).astype(np.float32)

            centroid_camera = np.median(
                xyz_camera,
                axis=0,
            ).astype(np.float64)
            centroid_base = np.median(
                xyz_base,
                axis=0,
            ).astype(np.float64)

            target_median_depth_m = float(
                np.median(xyz_camera[:, 2])
            )

            (
                dense_scene_xyz_camera,
                dense_scene_xyz_base,
                dense_scene_rgb,
                scene_xyz_camera,
                scene_xyz_base,
                scene_rgb,
                scene_stats,
            ) = self.build_scene_clouds(
                image_bgr=image_bgr,
                depth_m=depth_m,
                camera_info=camera_info,
                target_mask=processed_mask,
                target_bbox_xyxy=det.get("bbox_xyxy"),
                target_median_depth_m=target_median_depth_m,
                T_base_camera=T_base_camera,
            )

            cloud_ms = (time.perf_counter() - t_cloud) * 1000.0

            # Publish clouds.
            # IMPORTANT: create independent Header messages. Do not alias
            # color_msg.header, otherwise changing base_header.frame_id can also
            # change cam_header.frame_id in Python.
            cam_header = Header()
            cam_header.stamp = color_msg.header.stamp
            cam_header.frame_id = self.camera_frame

            base_header = Header()
            base_header.stamp = color_msg.header.stamp
            base_header.frame_id = self.base_frame

            cloud_cam_msg = point_cloud2.create_cloud_xyz32(
                cam_header,
                xyz_camera.tolist(),
            )
            cloud_base_msg = point_cloud2.create_cloud_xyz32(
                base_header,
                xyz_base.tolist(),
            )

            self.cloud_camera_pub.publish(cloud_cam_msg)
            self.cloud_base_pub.publish(cloud_base_msg)

            scene_cloud_cam_msg = point_cloud2.create_cloud_xyz32(
                cam_header,
                scene_xyz_camera.tolist(),
            )
            scene_cloud_base_msg = point_cloud2.create_cloud_xyz32(
                base_header,
                scene_xyz_base.tolist(),
            )

            self.scene_cloud_camera_pub.publish(scene_cloud_cam_msg)
            self.scene_cloud_base_pub.publish(scene_cloud_base_msg)

            dense_scene_cloud_cam_msg = point_cloud2.create_cloud_xyz32(
                cam_header,
                dense_scene_xyz_camera.tolist(),
            )
            dense_scene_cloud_base_msg = point_cloud2.create_cloud_xyz32(
                base_header,
                dense_scene_xyz_base.tolist(),
            )

            self.dense_scene_cloud_camera_pub.publish(
                dense_scene_cloud_cam_msg
            )
            self.dense_scene_cloud_base_pub.publish(
                dense_scene_cloud_base_msg
            )

            # Precise 3-D centroid.
            pose = PoseStamped()
            pose.header = base_header
            pose.pose.position.x = float(centroid_base[0])
            pose.pose.position.y = float(centroid_base[1])
            pose.pose.position.z = float(centroid_base[2])
            pose.pose.orientation.w = 1.0
            self.centroid_pub.publish(pose)

            marker = Marker()
            marker.header = base_header
            marker.ns = "open_vocab_local"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = 0.025
            marker.scale.y = 0.025
            marker.scale.z = 0.025
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.color.a = 1.0
            self.marker_pub.publish(marker)

            self.publish_debug(
                color_msg=color_msg,
                image_bgr=image_bgr,
                det=det,
                mask=processed_mask,
            )

            # Persist NPZ for Stage 2C / GraspNet.
            stamp_ns = (
                int(color_msg.header.stamp.sec) * 1_000_000_000
                + int(color_msg.header.stamp.nanosec)
            )
            safe_label = "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_"
                for ch in label
            )

            cloud_name = f"{stamp_ns}_{safe_label}_object_cloud.npz"
            cloud_path = self.output_dir / cloud_name

            K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)

            np.savez_compressed(
                cloud_path,
                label=np.array(label),
                object_id=np.array(str(scene.object_id)),
                stamp_ns=np.array(stamp_ns, dtype=np.int64),
                frame_camera=np.array(self.camera_frame),
                frame_base=np.array(self.base_frame),
                K=K,
                points_camera=xyz_camera,
                points_base=xyz_base,
                colors_rgb=rgb,
                centroid_camera=centroid_camera,
                centroid_base=centroid_base,
                confidence=np.array(
                    float(det.get("confidence", 0.0)),
                    dtype=np.float32,
                ),
                raw_mask_pixels=np.array(
                    depth_stats["raw_mask_pixels"], dtype=np.int64
                ),
                processed_mask_pixels=np.array(
                    depth_stats["processed_mask_pixels"], dtype=np.int64
                ),
                mask_depth_samples=np.array(
                    depth_stats["mask_depth_samples"], dtype=np.int64
                ),
                valid_depth_samples=np.array(
                    depth_stats["valid_depth_samples"], dtype=np.int64
                ),
                valid_depth_ratio=np.array(
                    depth_stats["valid_depth_ratio"], dtype=np.float32
                ),

                # Dense scene for GraspNet NETWORK inference.
                dense_scene_points_camera=dense_scene_xyz_camera,
                dense_scene_points_base=dense_scene_xyz_base,
                dense_scene_colors_rgb=dense_scene_rgb,

                # Downsampled scene for collision/debug (backward-compatible).
                scene_points_camera=scene_xyz_camera,
                scene_points_base=scene_xyz_base,
                scene_colors_rgb=scene_rgb,

                T_base_camera=T_base_camera.astype(np.float64),
                table_z=np.array(self.table_z, dtype=np.float64),
            )

            latest_npz = self.output_dir / "latest_object_cloud.npz"
            # np.savez_compressed cannot atomically replace by itself.
            tmp_npz = self.output_dir / "latest_object_cloud_tmp.npz"
            np.savez_compressed(
                tmp_npz,
                label=np.array(label),
                object_id=np.array(str(scene.object_id)),
                stamp_ns=np.array(stamp_ns, dtype=np.int64),
                frame_camera=np.array(self.camera_frame),
                frame_base=np.array(self.base_frame),
                K=K,
                points_camera=xyz_camera,
                points_base=xyz_base,
                colors_rgb=rgb,
                centroid_camera=centroid_camera,
                centroid_base=centroid_base,
                confidence=np.array(
                    float(det.get("confidence", 0.0)),
                    dtype=np.float32,
                ),
                raw_mask_pixels=np.array(
                    depth_stats["raw_mask_pixels"], dtype=np.int64
                ),
                processed_mask_pixels=np.array(
                    depth_stats["processed_mask_pixels"], dtype=np.int64
                ),
                mask_depth_samples=np.array(
                    depth_stats["mask_depth_samples"], dtype=np.int64
                ),
                valid_depth_samples=np.array(
                    depth_stats["valid_depth_samples"], dtype=np.int64
                ),
                valid_depth_ratio=np.array(
                    depth_stats["valid_depth_ratio"], dtype=np.float32
                ),

                # Dense scene for GraspNet NETWORK inference.
                dense_scene_points_camera=dense_scene_xyz_camera,
                dense_scene_points_base=dense_scene_xyz_base,
                dense_scene_colors_rgb=dense_scene_rgb,

                # Downsampled scene for collision/debug (backward-compatible).
                scene_points_camera=scene_xyz_camera,
                scene_points_base=scene_xyz_base,
                scene_colors_rgb=scene_rgb,

                T_base_camera=T_base_camera.astype(np.float64),
                table_z=np.array(self.table_z, dtype=np.float64),
            )
            tmp_npz.replace(latest_npz)

            median_depth = float(
                np.median(xyz_camera[:, 2])
            )

            total_ms = (time.perf_counter() - t_total) * 1000.0

            result = {
                "status": "ok",
                "object_id": str(scene.object_id),
                "label": label,
                "confidence": float(det.get("confidence", 0.0)),
                "sam_score": float(det.get("sam_score", 0.0)),
                "bbox_xyxy": [
                    float(x) for x in det.get("bbox_xyxy", [])
                ],
                "centroid_uv": [
                    float(x) for x in det.get("centroid_uv", [])
                ],
                "camera_frame": self.camera_frame,
                "base_frame": self.base_frame,
                "centroid_camera_xyz_m": centroid_camera.tolist(),
                "centroid_base_xyz_m": centroid_base.tolist(),
                "median_depth_m": median_depth,
                "cloud_points": int(len(xyz_camera)),

                # Depth coverage diagnostics. These values tell us whether a
                # partial cloud is a normal single-view surface or a D435i
                # depth-hole problem.
                "raw_mask_pixels": int(
                    depth_stats["raw_mask_pixels"]
                ),
                "processed_mask_pixels": int(
                    depth_stats["processed_mask_pixels"]
                ),
                "mask_depth_samples": int(
                    depth_stats["mask_depth_samples"]
                ),
                "valid_depth_samples": int(
                    depth_stats["valid_depth_samples"]
                ),
                "valid_depth_ratio": float(
                    depth_stats["valid_depth_ratio"]
                ),
                "post_percentile_depth_samples": int(
                    depth_stats["post_percentile_depth_samples"]
                ),
                "pre_voxel_points": int(
                    depth_stats["pre_voxel_points"]
                ),
                "post_voxel_points": int(
                    depth_stats["post_voxel_points"]
                ),

                "dense_scene_points": int(
                    scene_stats["dense_scene_points"]
                ),
                "dense_scene_points_before_cap": int(
                    scene_stats["dense_scene_points_before_cap"]
                ),
                "dense_scene_pixel_stride": int(
                    scene_stats["dense_scene_pixel_stride"]
                ),
                "collision_scene_points": int(
                    scene_stats["collision_scene_points"]
                ),
                # Backward-compatible alias.
                "scene_points": int(
                    scene_stats["collision_scene_points"]
                ),
                "scene_roi_xyxy": scene_stats["scene_roi_xyxy"],
                "scene_depth_near_m": float(
                    scene_stats["scene_depth_near_m"]
                ),
                "scene_depth_far_m": float(
                    scene_stats["scene_depth_far_m"]
                ),
                "table_z_m": float(self.table_z),

                "cloud_npz_path": str(cloud_path),
                "latest_cloud_npz_path": str(latest_npz),
                "latency_ms": {
                    "decode_ms": decode_ms,
                    "a6000_roundtrip_ms": ai_roundtrip_ms,
                    "cloud_and_tf_ms": cloud_ms,
                    "total_ms": total_ms,
                },
            }

            result_msg = String()
            result_msg.data = json.dumps(
                result,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self.result_pub.publish(result_msg)

            result_json_path = self.output_dir / "latest_local_result.json"
            tmp_json_path = self.output_dir / "latest_local_result.tmp"

            with open(
                tmp_json_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            tmp_json_path.replace(result_json_path)

            message = (
                f"Local perception OK: {label}, "
                f"points={len(xyz_camera)}, "
                f"centroid_base="
                f"({centroid_base[0]:.4f}, "
                f"{centroid_base[1]:.4f}, "
                f"{centroid_base[2]:.4f}) m, "
                f"median_depth={median_depth:.4f} m, "
                f"valid_depth_ratio="
                f"{100.0 * depth_stats['valid_depth_ratio']:.1f}%, "
                f"dense_scene={scene_stats['dense_scene_points']}, "
                f"collision_scene={scene_stats['collision_scene_points']}, "
                f"total={total_ms:.1f} ms"
            )

            return True, message

        except Exception as exc:
            # rclpy Humble's RcutilsLogger has no .exception() method.
            # Log the full traceback explicitly, then return a normal Trigger
            # failure response so the node stays alive.
            self.get_logger().error(
                "Local perception failed:\n"
                + traceback.format_exc()
            )
            return False, (
                f"{type(exc).__name__}: {exc}"
            )


def main():
    rclpy.init()

    node = WristRGBDLocalizer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
