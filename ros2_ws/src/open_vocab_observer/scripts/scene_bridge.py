#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from open_vocab_interfaces.msg import SceneObject


class SceneBridge(Node):
    """
    Bridges the non-ROS Stage-2A scene store into ROS2.

    It watches robot_host_stage2a/outputs/latest_scene.json and publishes only
    when a new object_id appears. This keeps perception independent from ROS2
    while giving the motion stack a typed, latched SceneObject message.
    """

    def __init__(self) -> None:
        super().__init__('open_vocab_scene_bridge')

        self.declare_parameter('scene_file', '')
        self.declare_parameter('poll_period_s', 0.20)
        self.declare_parameter('publish_on_start', False)
        self.declare_parameter('required_base_frame', 'base_link')

        scene_file = str(self.get_parameter('scene_file').value).strip()
        if not scene_file:
            raise RuntimeError('scene_file parameter must be configured')

        self.scene_file = Path(scene_file).expanduser().resolve()
        self.poll_period_s = float(self.get_parameter('poll_period_s').value)
        self.publish_on_start = bool(self.get_parameter('publish_on_start').value)
        self.required_base_frame = str(self.get_parameter('required_base_frame').value)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.publisher = self.create_publisher(
            SceneObject,
            '/open_vocab/scene_object',
            qos,
        )

        self._last_mtime_ns: Optional[int] = None
        self._last_object_id: Optional[str] = None
        self._first_poll = True

        self.timer = self.create_timer(self.poll_period_s, self._poll)

        self.get_logger().info(f'Watching scene file: {self.scene_file}')

    def _poll(self) -> None:
        if not self.scene_file.exists():
            return

        try:
            stat = self.scene_file.stat()
        except OSError as exc:
            self.get_logger().warning(f'stat failed: {exc}')
            return

        if self._last_mtime_ns == stat.st_mtime_ns:
            return
        self._last_mtime_ns = stat.st_mtime_ns

        try:
            with self.scene_file.open('r', encoding='utf-8') as f:
                payload = json.load(f)
            obj = self._select_newest(payload)
            self._validate_object(obj)
        except Exception as exc:
            self.get_logger().error(f'Invalid scene file: {exc}')
            return

        object_id = str(obj['object_id'])

        if self._first_poll:
            self._first_poll = False
            if not self.publish_on_start:
                self._last_object_id = object_id
                return

        if object_id == self._last_object_id:
            return

        msg = self._to_msg(obj)
        self.publisher.publish(msg)
        self._last_object_id = object_id

        p = obj['coarse_base_xy']
        self.get_logger().info(
            f"Published {obj['label']} id={object_id[:8]} "
            f"XY=({float(p[0]):.4f}, {float(p[1]):.4f}) "
            f"conf={float(obj['confidence']):.3f}"
        )

    @staticmethod
    def _select_newest(payload: Dict[str, Any]) -> Dict[str, Any]:
        objects = payload.get('objects')
        if not isinstance(objects, list) or not objects:
            raise ValueError('latest_scene.json contains no objects')
        return max(objects, key=lambda item: float(item.get('timestamp', 0.0)))

    def _validate_object(self, obj: Dict[str, Any]) -> None:
        required = [
            'object_id', 'query', 'label', 'role', 'confidence', 'frame_id',
            'timestamp', 'image_space_id', 'bbox_xyxy', 'primary_uv',
            'primary_point_type', 'coarse_base_xy', 'coarse_base_frame',
            'has_precise_3d_pose',
        ]
        missing = [k for k in required if k not in obj]
        if missing:
            raise ValueError(f'missing fields: {missing}')

        if str(obj['coarse_base_frame']) != self.required_base_frame:
            raise ValueError(
                f"coarse_base_frame={obj['coarse_base_frame']!r}, "
                f"expected {self.required_base_frame!r}"
            )

        xy = obj['coarse_base_xy']
        if not isinstance(xy, list) and not isinstance(xy, tuple):
            raise ValueError('coarse_base_xy must be a 2-element array')
        if len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy):
            raise ValueError(f'invalid coarse_base_xy: {xy}')

    def _to_msg(self, obj: Dict[str, Any]) -> SceneObject:
        msg = SceneObject()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(obj['coarse_base_frame'])

        msg.object_id = str(obj['object_id'])
        msg.query = str(obj['query'])
        msg.label = str(obj['label'])
        msg.role = str(obj['role'])
        msg.confidence = float(obj['confidence'])

        sam_score = obj.get('sam_score')
        msg.sam_score = float(sam_score) if sam_score is not None else float('nan')

        msg.source_frame_id = int(obj['frame_id'])
        msg.source_timestamp = float(obj['timestamp'])
        msg.image_space_id = str(obj['image_space_id'])

        msg.bbox_xyxy = [float(v) for v in obj['bbox_xyxy']]
        msg.centroid_uv = self._point_or_nan(obj.get('centroid_uv'))
        msg.support_uv = self._point_or_nan(obj.get('support_uv'))
        msg.primary_uv = [float(v) for v in obj['primary_uv']]
        msg.primary_point_type = str(obj['primary_point_type'])

        msg.coarse_position.x = float(obj['coarse_base_xy'][0])
        msg.coarse_position.y = float(obj['coarse_base_xy'][1])
        # Z is intentionally unknown at global-perception stage.
        msg.coarse_position.z = float('nan')

        msg.has_precise_3d_pose = bool(obj['has_precise_3d_pose'])
        msg.debug_image_url = str(obj.get('debug_image_url') or '')
        return msg

    @staticmethod
    def _point_or_nan(value: Any):
        if value is None:
            return [float('nan'), float('nan')]
        return [float(value[0]), float(value[1])]


def main(args=None):
    rclpy.init(args=args)
    node = SceneBridge()
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
