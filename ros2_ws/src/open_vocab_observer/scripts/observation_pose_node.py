#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from open_vocab_interfaces.msg import SceneObject


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError('zero quaternion')
    q /= n
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(m))

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def transform_matrix(translation, quaternion) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(*quaternion)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def matrix_to_pose(T: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])
    qx, qy, qz, qw = rot_to_quat(T[:3, :3])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def rotation_angle_deg(R: np.ndarray) -> float:
    value = (float(np.trace(R)) - 1.0) * 0.5
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def look_at_camera_transform(
    camera_xyz: np.ndarray,
    target_xyz: np.ndarray,
    preferred_x_base: np.ndarray,
) -> np.ndarray:
    """
    Build desired T_base_camera_optical.

    RealSense optical frame convention:
      +X = image right
      +Y = image down
      +Z = optical forward

    +Z is aimed exactly at target. The optical +X direction is chosen by
    projecting the CURRENT camera +X direction into the new image plane.
    This preserves camera roll as much as possible and reduces unnecessary
    wrist rotation between successive observations.
    """
    camera_xyz = np.asarray(camera_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    preferred_x_base = np.asarray(preferred_x_base, dtype=np.float64)

    z_axis = target_xyz - camera_xyz
    distance = float(np.linalg.norm(z_axis))
    if distance < 1e-6:
        raise ValueError('camera position equals focus point')
    z_axis /= distance

    x_axis = preferred_x_base - np.dot(preferred_x_base, z_axis) * z_axis

    if float(np.linalg.norm(x_axis)) < 1e-5:
        for fallback in (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ):
            x_axis = fallback - np.dot(fallback, z_axis) * z_axis
            if float(np.linalg.norm(x_axis)) >= 1e-5:
                break

    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = camera_xyz
    return T


class ObservationPoseNode(Node):
    """
    Convert coarse target XY into candidate poses for a RIGID observation link.

    IMPORTANT:
      The D435i is physically mounted on `observation_link`
      (for this robot: spherical_wrist_2_link), not on the grasp TCP.

    Geometry:
      T_base_camera_desired
        = T_base_observation_desired * T_observation_camera

      therefore

      T_base_observation_desired
        = T_base_camera_desired * inverse(T_observation_camera)

    MoveIt must constrain the SAME observation_link.
    """

    def __init__(self) -> None:
        super().__init__('open_vocab_observation_pose')

        self.declare_parameter('geometry_configured', False)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('observation_link', 'spherical_wrist_2_link')
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('table_z', 0.0)
        self.declare_parameter('focus_z_offset', 0.03)
        self.declare_parameter('candidate_offsets', [
            0.00,  0.00, 0.50,
           -0.12,  0.00, 0.48,
            0.12,  0.00, 0.48,
            0.00, -0.12, 0.48,
            0.00,  0.12, 0.48,
        ])
        self.declare_parameter('min_camera_height_above_table', 0.30)
        self.declare_parameter('tf_timeout_s', 1.0)
        self.declare_parameter('tf_retry_period_s', 0.25)
        self.declare_parameter('rigid_tf_translation_tolerance_m', 0.002)
        self.declare_parameter('rigid_tf_rotation_tolerance_deg', 1.0)
        self.declare_parameter('lookat_validation_min_dot', 0.995)

        self.geometry_configured = bool(self.get_parameter('geometry_configured').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.observation_link = str(self.get_parameter('observation_link').value)
        self.camera_frame = str(self.get_parameter('camera_optical_frame').value)
        self.table_z = float(self.get_parameter('table_z').value)
        self.focus_z_offset = float(self.get_parameter('focus_z_offset').value)
        self.min_camera_height = float(self.get_parameter('min_camera_height_above_table').value)
        self.tf_timeout_s = float(self.get_parameter('tf_timeout_s').value)
        self.tf_retry_period_s = float(self.get_parameter('tf_retry_period_s').value)
        self.rigid_tf_translation_tolerance_m = float(
            self.get_parameter('rigid_tf_translation_tolerance_m').value
        )
        self.rigid_tf_rotation_tolerance_deg = float(
            self.get_parameter('rigid_tf_rotation_tolerance_deg').value
        )
        self.lookat_validation_min_dot = float(
            self.get_parameter('lookat_validation_min_dot').value
        )

        flat_offsets = [float(v) for v in self.get_parameter('candidate_offsets').value]
        if len(flat_offsets) == 0 or len(flat_offsets) % 3 != 0:
            raise RuntimeError('candidate_offsets must contain 3*N values')

        self.offsets = [
            np.array(flat_offsets[i:i+3], dtype=np.float64)
            for i in range(0, len(flat_offsets), 3)
        ]

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.subscription = self.create_subscription(
            SceneObject,
            '/open_vocab/scene_object',
            self._on_scene_object,
            qos,
        )

        self.observation_pose_pub = self.create_publisher(
            PoseArray,
            '/open_vocab/observation_candidates',
            qos,
        )
        self.camera_pose_pub = self.create_publisher(
            PoseArray,
            '/open_vocab/camera_candidates',
            qos,
        )
        self.target_pub = self.create_publisher(
            PointStamped,
            '/open_vocab/observation_target',
            qos,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/open_vocab/observation_markers',
            qos,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._pending_msg: Optional[SceneObject] = None
        self._last_object_id: Optional[str] = None
        self._reference_T_observation_camera: Optional[np.ndarray] = None
        self._last_tf_warning_time = 0.0

        self.retry_timer = self.create_timer(
            self.tf_retry_period_s,
            self._retry_pending,
        )

        if not self.geometry_configured:
            self.get_logger().error(
                'geometry_configured=false. Fill the REAL table_z and frame names '
                'in observer.yaml before using candidate poses.'
            )

        self.get_logger().info(
            'ObservationPoseNode '
            f'base={self.base_frame} '
            f'observation_link={self.observation_link} '
            f'camera={self.camera_frame}'
        )

    def _on_scene_object(self, msg: SceneObject) -> None:
        if msg.object_id == self._last_object_id:
            return

        self._pending_msg = msg
        self._try_process_pending()

    def _retry_pending(self) -> None:
        if self._pending_msg is not None:
            self._try_process_pending()

    def _try_process_pending(self) -> None:
        msg = self._pending_msg
        if msg is None:
            return

        if not self.geometry_configured:
            return

        if msg.header.frame_id != self.base_frame:
            self.get_logger().error(
                f'SceneObject frame {msg.header.frame_id!r} != {self.base_frame!r}'
            )
            self._pending_msg = None
            return

        try:
            T_observation_camera = self._lookup_transform_matrix(
                self.observation_link,
                self.camera_frame,
            )
            T_base_camera_current = self._lookup_transform_matrix(
                self.base_frame,
                self.camera_frame,
            )
        except TransformException as exc:
            now = time.monotonic()
            if now - self._last_tf_warning_time > 3.0:
                self.get_logger().warning(
                    'Waiting for TF '
                    f'{self.observation_link} -> {self.camera_frame}: {exc}'
                )
                self._last_tf_warning_time = now
            return
        except Exception as exc:
            self.get_logger().error(f'TF lookup failed: {exc}')
            return

        try:
            self._validate_rigid_camera_mount(T_observation_camera)
        except Exception as exc:
            self.get_logger().error(
                f'Observation-link/camera TF is not rigid: {exc}. '
                'Refusing candidate generation.'
            )
            return

        target = np.array([
            float(msg.coarse_position.x),
            float(msg.coarse_position.y),
            self.table_z + self.focus_z_offset,
        ], dtype=np.float64)

        current_camera_x = T_base_camera_current[:3, 0].copy()
        T_camera_observation = np.linalg.inv(T_observation_camera)

        observation_poses: List[Pose] = []
        camera_poses: List[Pose] = []
        camera_positions: List[np.ndarray] = []

        for idx, offset in enumerate(self.offsets):
            camera_xyz = np.array([
                target[0] + offset[0],
                target[1] + offset[1],
                self.table_z + offset[2],
            ], dtype=np.float64)

            if camera_xyz[2] < self.table_z + self.min_camera_height:
                self.get_logger().warning(
                    f'Candidate {idx} rejected: camera Z={camera_xyz[2]:.3f} too low'
                )
                continue

            try:
                T_base_camera_desired = look_at_camera_transform(
                    camera_xyz=camera_xyz,
                    target_xyz=target,
                    preferred_x_base=current_camera_x,
                )

                T_base_observation_desired = (
                    T_base_camera_desired @ T_camera_observation
                )

                self._validate_candidate_geometry(
                    idx=idx,
                    T_base_observation=T_base_observation_desired,
                    T_observation_camera=T_observation_camera,
                    T_base_camera_desired=T_base_camera_desired,
                    target=target,
                )

            except Exception as exc:
                self.get_logger().warning(
                    f'Candidate {idx} geometry failed: {exc}'
                )
                continue

            observation_poses.append(
                matrix_to_pose(T_base_observation_desired)
            )
            camera_poses.append(
                matrix_to_pose(T_base_camera_desired)
            )
            camera_positions.append(camera_xyz)

        if not observation_poses:
            self.get_logger().error('No valid observation candidates generated')
            return

        now = self.get_clock().now().to_msg()

        observation_array = PoseArray()
        observation_array.header.stamp = now
        observation_array.header.frame_id = self.base_frame
        observation_array.poses = observation_poses
        self.observation_pose_pub.publish(observation_array)

        camera_array = PoseArray()
        camera_array.header = observation_array.header
        camera_array.poses = camera_poses
        self.camera_pose_pub.publish(camera_array)

        target_msg = PointStamped()
        target_msg.header = observation_array.header
        target_msg.point.x = float(target[0])
        target_msg.point.y = float(target[1])
        target_msg.point.z = float(target[2])
        self.target_pub.publish(target_msg)

        self.marker_pub.publish(
            self._make_markers(
                scene_msg=msg,
                target=target,
                observation_poses=observation_poses,
                camera_poses=camera_poses,
                stamp=now,
            )
        )

        self._last_object_id = msg.object_id
        self._pending_msg = None

        self.get_logger().info(
            f'Generated {len(observation_poses)} observation-link candidates for '
            f'{msg.label} target=({target[0]:.3f},{target[1]:.3f},{target[2]:.3f}) '
            f'link={self.observation_link}'
        )

    def _lookup_transform_matrix(self, target_frame: str, source_frame: str) -> np.ndarray:
        tf = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_timeout_s),
        )
        t = tf.transform.translation
        q = tf.transform.rotation
        return transform_matrix(
            [t.x, t.y, t.z],
            [q.x, q.y, q.z, q.w],
        )

    def _validate_rigid_camera_mount(self, T_observation_camera: np.ndarray) -> None:
        if self._reference_T_observation_camera is None:
            self._reference_T_observation_camera = T_observation_camera.copy()
            trans = T_observation_camera[:3, 3]
            self.get_logger().info(
                'Captured rigid camera mount TF '
                f'{self.observation_link}->{self.camera_frame}: '
                f't=({trans[0]:.4f},{trans[1]:.4f},{trans[2]:.4f})'
            )
            return

        T_ref = self._reference_T_observation_camera
        translation_error = float(
            np.linalg.norm(T_observation_camera[:3, 3] - T_ref[:3, 3])
        )
        R_delta = T_ref[:3, :3].T @ T_observation_camera[:3, :3]
        rotation_error_deg = rotation_angle_deg(R_delta)

        if translation_error > self.rigid_tf_translation_tolerance_m:
            raise RuntimeError(
                f'translation changed by {translation_error:.6f} m '
                f'(limit {self.rigid_tf_translation_tolerance_m:.6f} m)'
            )
        if rotation_error_deg > self.rigid_tf_rotation_tolerance_deg:
            raise RuntimeError(
                f'rotation changed by {rotation_error_deg:.3f} deg '
                f'(limit {self.rigid_tf_rotation_tolerance_deg:.3f} deg)'
            )

    def _validate_candidate_geometry(
        self,
        idx: int,
        T_base_observation: np.ndarray,
        T_observation_camera: np.ndarray,
        T_base_camera_desired: np.ndarray,
        target: np.ndarray,
    ) -> None:
        # Closed-loop transform consistency check.
        T_reconstructed_camera = T_base_observation @ T_observation_camera

        position_error = float(np.linalg.norm(
            T_reconstructed_camera[:3, 3] - T_base_camera_desired[:3, 3]
        ))
        rotation_error_deg = rotation_angle_deg(
            T_base_camera_desired[:3, :3].T
            @ T_reconstructed_camera[:3, :3]
        )

        if position_error > 1e-6 or rotation_error_deg > 1e-4:
            raise RuntimeError(
                f'camera reconstruction mismatch: pos={position_error:.3e} m, '
                f'rot={rotation_error_deg:.3e} deg'
            )

        camera_position = T_reconstructed_camera[:3, 3]
        camera_forward = T_reconstructed_camera[:3, 2]
        to_target = target - camera_position
        distance = float(np.linalg.norm(to_target))
        if distance < 1e-6:
            raise RuntimeError('camera target distance is too small')
        to_target /= distance

        dot = float(np.dot(camera_forward, to_target))
        if dot < self.lookat_validation_min_dot:
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            raise RuntimeError(
                f'look-at validation failed: dot={dot:.6f}, angle={angle:.3f} deg'
            )

        self.get_logger().debug(
            f'Candidate {idx}: desired camera look-at dot={dot:.6f}'
        )

    def _make_markers(
        self,
        scene_msg: SceneObject,
        target: np.ndarray,
        observation_poses: List[Pose],
        camera_poses: List[Pose],
        stamp,
    ) -> MarkerArray:
        markers = MarkerArray()

        target_marker = Marker()
        target_marker.header.frame_id = self.base_frame
        target_marker.header.stamp = stamp
        target_marker.ns = 'open_vocab_target'
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = float(target[0])
        target_marker.pose.position.y = float(target[1])
        target_marker.pose.position.z = float(target[2])
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.04
        target_marker.scale.y = 0.04
        target_marker.scale.z = 0.04
        target_marker.color.r = 1.0
        target_marker.color.g = 0.2
        target_marker.color.b = 0.1
        target_marker.color.a = 0.95
        markers.markers.append(target_marker)

        text_marker = Marker()
        text_marker.header = target_marker.header
        text_marker.ns = 'open_vocab_target_text'
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = float(target[0])
        text_marker.pose.position.y = float(target[1])
        text_marker.pose.position.z = float(target[2] + 0.06)
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.04
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = scene_msg.label
        markers.markers.append(text_marker)

        for i, (obs_pose, cam_pose) in enumerate(zip(observation_poses, camera_poses)):
            obs_marker = Marker()
            obs_marker.header.frame_id = self.base_frame
            obs_marker.header.stamp = stamp
            obs_marker.ns = 'open_vocab_observation_link_candidates'
            obs_marker.id = 100 + i
            obs_marker.type = Marker.ARROW
            obs_marker.action = Marker.ADD
            obs_marker.pose = obs_pose
            obs_marker.scale.x = 0.10
            obs_marker.scale.y = 0.018
            obs_marker.scale.z = 0.018
            obs_marker.color.r = 0.1
            obs_marker.color.g = 0.7
            obs_marker.color.b = 1.0
            obs_marker.color.a = 0.9
            markers.markers.append(obs_marker)

            cam_sphere = Marker()
            cam_sphere.header = obs_marker.header
            cam_sphere.ns = 'open_vocab_camera_candidates'
            cam_sphere.id = 200 + i
            cam_sphere.type = Marker.SPHERE
            cam_sphere.action = Marker.ADD
            cam_sphere.pose.position = cam_pose.position
            cam_sphere.pose.orientation.w = 1.0
            cam_sphere.scale.x = 0.025
            cam_sphere.scale.y = 0.025
            cam_sphere.scale.z = 0.025
            cam_sphere.color.r = 0.8
            cam_sphere.color.g = 0.3
            cam_sphere.color.b = 1.0
            cam_sphere.color.a = 0.95
            markers.markers.append(cam_sphere)

            # Green ray: desired optical +Z direction, from camera to target.
            ray = Marker()
            ray.header = obs_marker.header
            ray.ns = 'open_vocab_camera_look_rays'
            ray.id = 300 + i
            ray.type = Marker.ARROW
            ray.action = Marker.ADD
            ray.scale.x = 0.012  # shaft diameter
            ray.scale.y = 0.022  # head diameter
            ray.scale.z = 0.035  # head length
            ray.color.r = 0.1
            ray.color.g = 1.0
            ray.color.b = 0.2
            ray.color.a = 0.95

            p0 = Point()
            p0.x = float(cam_pose.position.x)
            p0.y = float(cam_pose.position.y)
            p0.z = float(cam_pose.position.z)
            p1 = Point()
            p1.x = float(target[0])
            p1.y = float(target[1])
            p1.z = float(target[2])
            ray.points = [p0, p1]
            markers.markers.append(ray)

        return markers


def main(args=None):
    rclpy.init(args=args)
    node = ObservationPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
