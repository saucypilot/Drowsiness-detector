from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class DrowsinessResult:
    ear: Optional[float]
    closed_frames: int
    is_drowsy: bool
    left_eye: Optional[np.ndarray]
    right_eye: Optional[np.ndarray]


class DrowsinessDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        ear_threshold: float = 0.22,
        consec_frames: int = 20,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.closed_frames = 0
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> DrowsinessResult:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        ear = None
        left_pts = None
        right_pts = None

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            left_pts = np.array([self._lm_to_xy(lms[i], w, h) for i in self.LEFT_EYE], dtype=np.float32)
            right_pts = np.array([self._lm_to_xy(lms[i], w, h) for i in self.RIGHT_EYE], dtype=np.float32)

            ear = (self._eye_aspect_ratio(left_pts) + self._eye_aspect_ratio(right_pts)) / 2.0

            if ear < self.ear_threshold:
                self.closed_frames += 1
            else:
                self.closed_frames = 0

        is_drowsy = self.closed_frames >= self.consec_frames
        return DrowsinessResult(ear=ear, closed_frames=self.closed_frames, is_drowsy=is_drowsy,
                                left_eye=left_pts, right_eye=right_pts)

    def close(self) -> None:
        self._face_mesh.close()

    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = pts
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        return 0.0 if h == 0 else (v1 + v2) / (2.0 * h)

    @staticmethod
    def _lm_to_xy(lm, w: int, h: int) -> np.ndarray:
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
