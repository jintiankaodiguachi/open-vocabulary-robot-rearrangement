#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np


class CameraBackend(ABC):
    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def read(self) -> np.ndarray:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


def _first_existing_key(data: np.lib.npyio.NpzFile, keys):
    for key in keys:
        if key in data.files:
            return key
    return None


class HikExistingBackend(CameraBackend):
    """
    OpenCV/V4L2 HIK camera backend.

    Runtime output must be:
      - BGR uint8
      - shape: (720, 1280, 3)
      - same pixel coordinate space used by Homography calibration
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.device: Union[int, str] = cfg.get("device", 0)
        self.width = int(cfg.get("width", 1280))
        self.height = int(cfg.get("height", 720))
        self.fps = float(cfg.get("fps", 30))
        self.fourcc = str(cfg.get("fourcc", "MJPG"))
        self.use_v4l2 = bool(cfg.get("use_v4l2", True))
        self.do_undistort = bool(cfg.get("do_undistort", True))
        self.intrinsic_file = Path(str(cfg.get("intrinsic_file", "camera_intrinsics.npz")))
        self.new_camera_matrix_key = str(cfg.get("new_camera_matrix_key", "")).strip()
        self.warmup_frames = int(cfg.get("warmup_frames", 5))

        self.cap: Optional[cv2.VideoCapture] = None
        self.K = None
        self.dist = None
        self.newK = None
        self.map1 = None
        self.map2 = None

        if self.do_undistort:
            self._load_intrinsics()
            self._build_undistort_maps()

    def _load_intrinsics(self) -> None:
        if not self.intrinsic_file.exists():
            raise FileNotFoundError(
                f"Intrinsic file not found: {self.intrinsic_file.resolve()}"
            )

        data = np.load(str(self.intrinsic_file))

        k_key = _first_existing_key(data, ["K", "camera_matrix", "cameraMatrix", "mtx"])
        d_key = _first_existing_key(
            data,
            ["dist", "dist_coeffs", "distCoeffs", "distortion_coefficients"],
        )

        if k_key is None:
            raise KeyError(
                f"No camera matrix found. Available keys: {data.files}. "
                "Expected K/camera_matrix/cameraMatrix/mtx"
            )
        if d_key is None:
            raise KeyError(
                f"No distortion coefficients found. Available keys: {data.files}. "
                "Expected dist/dist_coeffs/distCoeffs/distortion_coefficients"
            )

        self.K = np.asarray(data[k_key], dtype=np.float64)
        self.dist = np.asarray(data[d_key], dtype=np.float64).reshape(-1)

        if self.K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {self.K.shape}")

        if self.new_camera_matrix_key:
            if self.new_camera_matrix_key not in data.files:
                raise KeyError(
                    f"new_camera_matrix_key='{self.new_camera_matrix_key}' not found. "
                    f"Available keys: {data.files}"
                )
            self.newK = np.asarray(data[self.new_camera_matrix_key], dtype=np.float64)
            if self.newK.shape != (3, 3):
                raise ValueError(
                    f"{self.new_camera_matrix_key} must be 3x3, got {self.newK.shape}"
                )
        else:
            self.newK = self.K.copy()

        print("[Camera] Loaded intrinsics")
        print(f"[Camera] intrinsic_file = {self.intrinsic_file.resolve()}")
        print(f"[Camera] K key = {k_key}")
        print(f"[Camera] dist key = {d_key}")
        print("[Camera] K =\n", self.K)
        print("[Camera] dist =\n", self.dist)
        print("[Camera] output camera matrix =\n", self.newK)

    def _build_undistort_maps(self) -> None:
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K,
            self.dist,
            None,
            self.newK,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        print(f"[Camera] Undistortion maps ready: {self.width}x{self.height}")

    def open(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return

        print(
            f"[Camera] Opening device={self.device}, "
            f"requested={self.width}x{self.height}@{self.fps}, fourcc={self.fourcc}"
        )

        if self.use_v4l2:
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.device)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device: {self.device}")

        if len(self.fourcc) == 4:
            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self.fourcc),
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"[Camera] Actual: {actual_width}x{actual_height}@{actual_fps:.2f}")

        if actual_width != self.width or actual_height != self.height:
            self.close()
            raise RuntimeError(
                f"Camera resolution mismatch: expected {self.width}x{self.height}, "
                f"got {actual_width}x{actual_height}. "
                "Do not resize silently because Homography depends on pixel coordinates."
            )

        for _ in range(max(0, self.warmup_frames)):
            ok, _ = self.cap.read()
            if not ok:
                time.sleep(0.03)

        print("[Camera] Camera ready")

    def read(self) -> np.ndarray:
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not open")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Expected 3-channel BGR image, got {frame.shape}")

        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            raise RuntimeError(
                f"Runtime frame size changed: got {w}x{h}, expected {self.width}x{self.height}"
            )

        if self.do_undistort:
            frame = cv2.remap(
                frame,
                self.map1,
                self.map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        h2, w2 = frame.shape[:2]
        if w2 != self.width or h2 != self.height:
            raise RuntimeError(
                f"Output frame must remain {self.width}x{self.height}, got {w2}x{h2}"
            )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(frame)

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
        print("[Camera] Closed")


def create_camera_backend(camera_cfg: Dict[str, Any]) -> CameraBackend:
    backend = str(camera_cfg.get("backend", "hik_existing")).strip().lower()
    if backend == "hik_existing":
        return HikExistingBackend(camera_cfg)
    raise ValueError(
        f"Unsupported camera backend: {backend}. Supported: hik_existing"
    )
