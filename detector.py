from dataclasses import dataclass
from collections import deque
from pathlib import Path
import time
from typing import Deque, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from head_pose import HeadPoseEstimator, HeadPoseAngles


@dataclass
class DrowsinessResult:
    ear: Optional[float]
    closed_duration_s: float
    risk_score: float
    risk_level: str
    perclos_percent: float
    blink_rate_per_min: float
    long_blinks_per_min: float
    recent_nods: int
    eye_status: str
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
        risk_window_s: float = 60.0,
        long_blink_min_s: float = 0.7,
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
        if risk_window_s <= 0:
            raise ValueError("risk_window_s must be positive")
        if long_blink_min_s <= 0:
            raise ValueError("long_blink_min_s must be positive")

        self.ear_threshold = ear_threshold
        self.drowsy_duration_s = drowsy_duration_s
        self.threshold_ratio = threshold_ratio
        self.risk_window_s = risk_window_s
        self.long_blink_min_s = long_blink_min_s
        self.baseline_ear: Optional[float] = None
        self.closed_since: Optional[float] = None
        self.closed_duration_s = 0.0
        self._eye_segments: Deque[Tuple[float, bool, float]] = deque()
        self._blink_events: Deque[Tuple[float, float]] = deque()
        self._nod_events: Deque[float] = deque()
        self._last_eye_sample_time: Optional[float] = None
        self._last_eyes_closed = False

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
        now = time.monotonic()
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
                eyes_closed = ear < self.ear_threshold
                self._update_eye_closure(eyes_closed, now=now)
                self._record_eye_sample(eyes_closed, now)
            else:
                self._update_eye_closure(False, now=now, record_event=False)
                self._reset_eye_sample_clock()

            angles, nod_detected = self._head_pose.estimate_from_landmarks(lms, w, h)
            pitch, yaw, roll = angles.pitch_deg, angles.yaw_deg, angles.roll_deg
            if nod_detected:
                self._nod_events.append(now)
        else:
            # Never count time when the face/eyes are not observable.
            self._update_eye_closure(False, now=now, record_event=False)
            self._reset_eye_sample_clock()

        risk_score, risk_level, perclos, blink_rate, long_blink_rate, recent_nods = (
            self._compute_risk(now)
        )
        is_drowsy = risk_score >= 61.0

        return DrowsinessResult(
            ear=ear,
            closed_duration_s=self.closed_duration_s,
            risk_score=risk_score,
            risk_level=risk_level,
            perclos_percent=perclos,
            blink_rate_per_min=blink_rate,
            long_blinks_per_min=long_blink_rate,
            recent_nods=recent_nods,
            eye_status=self._eye_status(long_blink_rate),
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
        self._reset_risk_state()
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
        record_event: bool = True,
    ) -> float:
        """Update and return continuous eye-closure time in seconds."""
        current_time = time.monotonic() if now is None else now
        if not eyes_closed:
            if record_event and self.closed_since is not None:
                duration_s = max(0.0, current_time - self.closed_since)
                if duration_s >= 0.08:
                    self._blink_events.append((current_time, duration_s))
            self.closed_since = None
            self.closed_duration_s = 0.0
            return self.closed_duration_s

        if self.closed_since is None:
            self.closed_since = current_time
        self.closed_duration_s = max(0.0, current_time - self.closed_since)
        return self.closed_duration_s

    def _record_eye_sample(self, eyes_closed: bool, now: float) -> None:
        if self._last_eye_sample_time is not None:
            duration_s = max(0.0, now - self._last_eye_sample_time)
            if duration_s > 0:
                self._eye_segments.append((now, self._last_eyes_closed, duration_s))

        self._last_eye_sample_time = now
        self._last_eyes_closed = eyes_closed
        self._prune_history(now)

    def _reset_eye_sample_clock(self) -> None:
        self._last_eye_sample_time = None
        self._last_eyes_closed = False

    def _reset_risk_state(self) -> None:
        self._update_eye_closure(False, record_event=False)
        self._eye_segments.clear()
        self._blink_events.clear()
        self._nod_events.clear()
        self._reset_eye_sample_clock()

    def _prune_history(self, now: float) -> None:
        cutoff = now - self.risk_window_s
        while self._eye_segments and self._eye_segments[0][0] < cutoff:
            self._eye_segments.popleft()
        while self._blink_events and self._blink_events[0][0] < cutoff:
            self._blink_events.popleft()
        while self._nod_events and self._nod_events[0] < cutoff:
            self._nod_events.popleft()

    def _compute_risk(
        self,
        now: float,
    ) -> Tuple[float, str, float, float, float, int]:
        self._prune_history(now)

        observed_s = sum(segment[2] for segment in self._eye_segments)
        closed_s = sum(segment[2] for segment in self._eye_segments if segment[1])
        perclos_percent = 0.0 if observed_s == 0 else (closed_s / observed_s) * 100.0

        minutes = max(observed_s / 60.0, 1.0 / 60.0)
        blink_count = len(self._blink_events)
        long_blink_count = sum(
            1 for _, duration_s in self._blink_events if duration_s >= self.long_blink_min_s
        )
        blink_rate = blink_count / minutes
        long_blink_rate = long_blink_count / minutes
        recent_nods = len(self._nod_events)

        prolonged_eye_closure = min(self.closed_duration_s / self.drowsy_duration_s, 1.0) * 100.0
        frequent_long_blinks = min(max(long_blink_count - 1, 0) / 5.0, 1.0) * 100.0
        nod_score = min(recent_nods / 4.0, 1.0) * 100.0
        perclos_score = min(perclos_percent / 40.0, 1.0) * 100.0

        risk_score = (
            prolonged_eye_closure * 0.45
            + frequent_long_blinks * 0.20
            + nod_score * 0.20
            + perclos_score * 0.15
        )
        if self.closed_duration_s >= self.drowsy_duration_s:
            risk_score = max(risk_score, 61.0)
        risk_score = min(100.0, max(0.0, risk_score))
        return (
            risk_score,
            self._risk_level(risk_score),
            perclos_percent,
            blink_rate,
            long_blink_rate,
            recent_nods,
        )

    def _eye_status(self, long_blinks_per_min: float) -> str:
        if self.closed_duration_s >= self.drowsy_duration_s:
            return "Prolonged closure"
        if self.closed_duration_s >= self.long_blink_min_s:
            return "Long blink"
        if long_blinks_per_min >= 6.0:
            return "Frequent closure"
        if self.closed_duration_s > 0:
            return "Closed"
        return "Open"

    @staticmethod
    def _risk_level(risk_score: float) -> str:
        if risk_score <= 30.0:
            return "ALERT"
        if risk_score <= 60.0:
            return "FATIGUE WARNING"
        if risk_score <= 80.0:
            return "DROWSY"
        return "CRITICAL"

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
