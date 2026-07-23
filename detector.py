from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from head_pose import HeadPoseEstimator, HeadPoseAngles


@dataclass
class DrowsinessResult:
    ear: Optional[float]
    closed_duration_s: float
    is_drowsy: bool
    left_eye: Optional[np.ndarray]
    right_eye: Optional[np.ndarray]
    pitch_deg: Optional[float]
    yaw_deg: Optional[float]
    roll_deg: Optional[float]
    nod_detected: bool


class DrowsinessDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        ear_threshold: Optional[float] = None,
        drowsy_duration_s: float = 1.5,
        threshold_ratio: float = 0.75,
        face_landmarker_model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if ear_threshold is not None and ear_threshold <= 0:
            raise ValueError("ear_threshold must be positive")
        if drowsy_duration_s <= 0:
            raise ValueError("drowsy_duration_s must be positive")
        if not 0 < threshold_ratio < 1:
            raise ValueError("threshold_ratio must be between 0 and 1")

        self.ear_threshold = ear_threshold
        self.drowsy_duration_s = drowsy_duration_s
        self.threshold_ratio = threshold_ratio
        self.baseline_ear: Optional[float] = None
        self.closed_since: Optional[float] = None
        self.closed_duration_s = 0.0

        self._uses_tasks_api = not hasattr(mp, "solutions")
        self._last_timestamp_ms = -1
        if self._uses_tasks_api:
            model_path = Path(
                face_landmarker_model_path
                or Path(__file__).resolve().parent / "models" / "face_landmarker.task"
            )
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"MediaPipe Face Landmarker model not found: {model_path}"
                )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        else:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

        self._head_pose = HeadPoseEstimator()

    def process(self, frame: np.ndarray) -> DrowsinessResult:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lms = self._detect_landmarks(rgb)

        ear = None
        left_pts = None
        right_pts = None

        pitch = yaw = roll = None
        nod_detected = False

        if lms is not None:
            left_pts = np.array([self._lm_to_xy(lms[i], w, h) for i in self.LEFT_EYE], dtype=np.float32)
            right_pts = np.array([self._lm_to_xy(lms[i], w, h) for i in self.RIGHT_EYE], dtype=np.float32)

            ear = (self._eye_aspect_ratio(left_pts) + self._eye_aspect_ratio(right_pts)) / 2.0

            # Do not classify eye closure until startup calibration has supplied
            # a personal threshold (or the caller explicitly provided one).
            if self.ear_threshold is not None:
                self._update_eye_closure(ear < self.ear_threshold)
            else:
                self._update_eye_closure(False)

            angles, nod_detected = self._head_pose.estimate_from_landmarks(lms, w, h)
            pitch, yaw, roll = angles.pitch_deg, angles.yaw_deg, angles.roll_deg
        else:
            # Never count time when the face/eyes are not observable.
            self._update_eye_closure(False)

        is_drowsy = self.closed_duration_s >= self.drowsy_duration_s

        return DrowsinessResult(
            ear=ear,
            closed_duration_s=self.closed_duration_s,
            is_drowsy=is_drowsy,
            left_eye=left_pts,
            right_eye=right_pts,
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            nod_detected=nod_detected,
        )

    @property
    def needs_calibration(self) -> bool:
        return self.ear_threshold is None

    def calibrate(self, samples: Iterable[float]) -> Tuple[float, float]:
        """Set a personal EAR threshold from normal open-eye observations."""
        values = np.asarray(list(samples), dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            raise ValueError("calibration requires at least one valid EAR sample")

        self.baseline_ear = float(np.median(values))
        self.ear_threshold = self.baseline_ear * self.threshold_ratio
        self._update_eye_closure(False)
        return self.baseline_ear, self.ear_threshold

    def close(self) -> None:
        self._face_mesh.close()

    def _detect_landmarks(self, rgb: np.ndarray):
        if not self._uses_tasks_api:
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0].landmark

        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._face_mesh.detect_for_video(image, timestamp_ms)
        if not results.face_landmarks:
            return None
        return results.face_landmarks[0]

    def _update_eye_closure(
        self,
        eyes_closed: bool,
        now: Optional[float] = None,
    ) -> float:
        """Update and return continuous eye-closure time in seconds."""
        if not eyes_closed:
            self.closed_since = None
            self.closed_duration_s = 0.0
            return self.closed_duration_s

        current_time = time.monotonic() if now is None else now
        if self.closed_since is None:
            self.closed_since = current_time
        self.closed_duration_s = max(0.0, current_time - self.closed_since)
        return self.closed_duration_s

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
