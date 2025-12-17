from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple
import time
import math

import cv2
import numpy as np


@dataclass
class HeadPoseAngles:
    pitch_deg: Optional[float]
    yaw_deg: Optional[float]
    roll_deg: Optional[float]


class NoddingDetector:
    """
    Simple pitch-based nod detector.
    Convention used here: positive pitch means "looking down" (typically true with this Euler extraction).
    """

    def __init__(
        self,
        down_threshold_deg: float = 15.0,
        up_threshold_deg: float = 8.0,
        max_interval_s: float = 1.5,
        cooldown_s: float = 1.0,
    ) -> None:
        self.down_threshold_deg = down_threshold_deg
        self.up_threshold_deg = up_threshold_deg
        self.max_interval_s = max_interval_s
        self.cooldown_s = cooldown_s

        self._state = "IDLE"  # IDLE -> DOWN -> (nod)
        self._down_time: Optional[float] = None
        self._last_fire: float = 0.0

    def update(self, pitch_deg: Optional[float], now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()

        if pitch_deg is None:
            # No measurement -> reset the gesture state (but keep cooldown)
            self._state = "IDLE"
            self._down_time = None
            return False

        # Cooldown gate
        if (now - self._last_fire) < self.cooldown_s:
            return False

        if self._state == "IDLE":
            if pitch_deg >= self.down_threshold_deg:
                self._state = "DOWN"
                self._down_time = now
            return False

        if self._state == "DOWN":
            assert self._down_time is not None
            # If it takes too long, abort
            if (now - self._down_time) > self.max_interval_s:
                self._state = "IDLE"
                self._down_time = None
                return False

            # Nod completion: after looking down, pitch returns "up-ish"
            if pitch_deg <= self.up_threshold_deg:
                self._state = "IDLE"
                self._down_time = None
                self._last_fire = now
                return True

        return False


class HeadPoseEstimator:
    """
    Head pose via solvePnP using a few MediaPipe FaceMesh landmark points.

    Landmarks used (MediaPipe FaceMesh indices):
      - Nose tip: 1
      - Chin: 152
      - Left eye outer corner: 33
      - Right eye outer corner: 263
      - Mouth left: 61
      - Mouth right: 291

    Returns Euler angles in degrees: pitch, yaw, roll.
    """

    # FaceMesh indices
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    def __init__(self) -> None:
        self.nod = NoddingDetector()

    def estimate_from_landmarks(
        self,
        lms,  # list[mediapipe.framework.formats.landmark_pb2.NormalizedLandmark]
        w: int,
        h: int,
    ) -> Tuple[HeadPoseAngles, bool]:
        """
        lms: MediaPipe landmark list (NormalizedLandmark)
        w,h: frame dimensions
        """
        image_points = self._get_image_points(lms, w, h)
        if image_points is None:
            return HeadPoseAngles(None, None, None), False

        # Generic 3D model points (rough human face model; scale doesn't matter much for angles)
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -63.6, -12.5),      # Chin
                (-43.3, 32.7, -26.0),     # Left eye outer corner
                (43.3, 32.7, -26.0),      # Right eye outer corner
                (-28.9, -28.9, -24.1),    # Mouth left
                (28.9, -28.9, -24.1),     # Mouth right
            ],
            dtype=np.float64,
        )

        # Camera intrinsics approximation
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return HeadPoseAngles(None, None, None), False

        pitch, yaw, roll = self._rvec_to_euler_deg(rvec)
        angles = HeadPoseAngles(pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)

        nod_detected = self.nod.update(pitch_deg=pitch)
        return angles, nod_detected

    def _get_image_points(self, lms, w: int, h: int) -> Optional[np.ndarray]:
        idxs = (
            self.NOSE_TIP,
            self.CHIN,
            self.LEFT_EYE_OUTER,
            self.RIGHT_EYE_OUTER,
            self.MOUTH_LEFT,
            self.MOUTH_RIGHT,
        )
        try:
            pts = []
            for i in idxs:
                lm = lms[i]
                pts.append((lm.x * w, lm.y * h))
            return np.array(pts, dtype=np.float64)
        except Exception:
            return None

    @staticmethod
    def _rvec_to_euler_deg(rvec: np.ndarray) -> Tuple[float, float, float]:
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Robust Euler extraction (assuming R = Rz(roll)*Ry(yaw)*Rx(pitch) style-ish)
        # We'll compute:
        # yaw   = atan2(r21, r11) ??? depends on convention; we’ll use a common CV decomposition:
        # pitch = atan2(-r31, sqrt(r11^2 + r21^2))
        # yaw   = atan2(r21, r11)
        # roll  = atan2(r32, r33)
        r11, r12, r13 = rmat[0, 0], rmat[0, 1], rmat[0, 2]
        r21, r22, r23 = rmat[1, 0], rmat[1, 1], rmat[1, 2]
        r31, r32, r33 = rmat[2, 0], rmat[2, 1], rmat[2, 2]

        # pitch
        pitch = math.atan2(-r31, math.sqrt(r11 * r11 + r21 * r21))
        # yaw
        yaw = math.atan2(r21, r11)
        # roll
        roll = math.atan2(r32, r33)

        return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))
