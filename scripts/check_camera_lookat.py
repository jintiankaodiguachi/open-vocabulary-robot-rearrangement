#!/usr/bin/env python3

import sys
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener


def quat_to_rotation_matrix(x, y, z, w):
    """
    Quaternion xyzw -> 3x3 rotation matrix
    """
    n = x*x + y*y + z*z + w*w

    if n < 1e-12:
        raise ValueError("Invalid zero quaternion")

    s = 2.0 / n

    xx = x * x * s
    yy = y * y * s
    zz = z * z * s

    xy = x * y * s
    xz = x * z * s
    yz = y * z * s

    wx = w * x * s
    wy = w * y * s
    wz = w * z * s

    return np.array([
        [
            1.0 - yy - zz,
            xy - wz,
            xz + wy,
        ],
        [
            xy + wz,
            1.0 - xx - zz,
            yz - wx,
        ],
        [
            xz - wy,
            yz + wx,
            1.0 - xx - yy,
        ]
    ], dtype=np.float64)


class CameraLookAtChecker(Node):

    def __init__(self, target):
        super().__init__("camera_lookat_checker")

        self.target = np.asarray(
            target,
            dtype=np.float64
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

    def check(self):

        # Give TF some time to arrive.
        for _ in range(30):

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            try:
                tf = self.tf_buffer.lookup_transform(
                    "base_link",
                    "camera_color_optical_frame",
                    rclpy.time.Time(),
                    timeout=Duration(seconds=1.0),
                )
                break

            except Exception as e:
                last_error = e

        else:
            raise RuntimeError(
                f"TF lookup failed: {last_error}"
            )

        t = tf.transform.translation
        q = tf.transform.rotation

        camera_position = np.array([
            t.x,
            t.y,
            t.z,
        ])

        R_base_camera = quat_to_rotation_matrix(
            q.x,
            q.y,
            q.z,
            q.w,
        )

        # Optical frame +Z = camera forward direction.
        camera_forward = R_base_camera[:, 2]

        to_target = (
            self.target
            - camera_position
        )

        distance = np.linalg.norm(to_target)

        if distance < 1e-6:
            raise RuntimeError(
                "Camera is already at target"
            )

        target_direction = (
            to_target / distance
        )

        dot = float(
            np.dot(
                camera_forward,
                target_direction
            )
        )

        dot_clamped = max(
            -1.0,
            min(1.0, dot)
        )

        angle_deg = math.degrees(
            math.acos(dot_clamped)
        )

        print()
        print("=" * 60)
        print("Camera Look-at Check")
        print("=" * 60)

        print(
            "target(base)       =",
            self.target
        )

        print(
            "camera_position    =",
            camera_position
        )

        print(
            "camera optical +Z  =",
            camera_forward
        )

        print(
            "camera -> target   =",
            target_direction
        )

        print(
            f"distance           = "
            f"{distance:.4f} m"
        )

        print(
            f"dot                = "
            f"{dot:.6f}"
        )

        print(
            f"angle error        = "
            f"{angle_deg:.2f} deg"
        )

        if dot > 0.95:
            print(
                "RESULT: GOOD - camera is "
                "looking at target."
            )

        elif dot > 0.7:
            print(
                "RESULT: WARNING - roughly toward "
                "target, but orientation error is large."
            )

        elif dot > 0.0:
            print(
                "RESULT: BAD - target is in front "
                "hemisphere but camera is poorly aimed."
            )

        else:
            print(
                "RESULT: WRONG - camera optical +Z "
                "points AWAY from target."
            )

        print("=" * 60)


def main():

    if len(sys.argv) != 4:
        print(
            "Usage:"
            "\n  python3 check_camera_lookat.py "
            "<target_x> <target_y> <target_z>"
        )
        sys.exit(1)

    target = [
        float(sys.argv[1]),
        float(sys.argv[2]),
        float(sys.argv[3]),
    ]

    rclpy.init()

    node = CameraLookAtChecker(
        target
    )

    try:
        node.check()

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
