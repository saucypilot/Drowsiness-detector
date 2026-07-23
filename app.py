import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from alarm import AlarmPlayer
from detector import DrowsinessDetector, DrowsinessResult
from window_utils import window_is_open


class DrowsinessDetectorApp:
    def __init__(
        self,
        camera_index: int = 0,
        window_name: str = "Drowsiness Detector",
        window_size: Tuple[int, int] = (960, 540),
        alarm_mode: str = "beep",
        alarm_wav_path: str = "alarm.wav",
        alarm_cooldown: float = 2.0,
        calibration_seconds: float = 5.0,
        calibration_min_samples: int = 15,
        threshold_ratio: float = 0.75,
        ear_threshold: Optional[float] = None,
        drowsy_duration_s: float = 1.5,
    ) -> None:
        if calibration_seconds <= 0:
            raise ValueError("calibration_seconds must be positive")
        if calibration_min_samples <= 0:
            raise ValueError("calibration_min_samples must be positive")

        self.camera_index = camera_index
        self.window_name = window_name
        self.window_size = window_size
        self.calibration_seconds = calibration_seconds
        self.calibration_min_samples = calibration_min_samples
        self.alarm = AlarmPlayer(mode=alarm_mode, wav_path=alarm_wav_path, cooldown=alarm_cooldown)
        self.detector = DrowsinessDetector(
            ear_threshold=ear_threshold,
            drowsy_duration_s=drowsy_duration_s,
            threshold_ratio=threshold_ratio,
        )

    def run(self) -> None:
        cap = self._open_camera()
        try:
            frame0 = self._read_initial_frame(cap)
            h0, w0 = frame0.shape[:2]
            self._log_camera_params(cap, w0, h0)
            self._setup_window()
            if self.detector.needs_calibration:
                frame0 = self._calibrate(cap, frame0)
                if frame0 is None:
                    return
            self._main_loop(cap, frame0)
        finally:
            cap.release()
            self.detector.close()
            cv2.destroyAllWindows()

    def _open_camera(self) -> cv2.VideoCapture:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("Cannot open camera")
            raise SystemExit
        return cap

    def _read_initial_frame(self, cap: cv2.VideoCapture) -> np.ndarray:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera read failed on startup")
            raise SystemExit
        return cv2.flip(frame, 1)

    def _log_camera_params(self, cap: cv2.VideoCapture, width: int, height: int) -> None:
        params = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height,
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
        }
        print("Camera params:", params)

    def _setup_window(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)

    def _main_loop(self, cap: cv2.VideoCapture, initial_frame: np.ndarray) -> None:
        frame = initial_frame
        while True:
            if not window_is_open(self.window_name):
                break

            result = self.detector.process(frame)
            self._draw_overlays(frame, result)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if not window_is_open(self.window_name) or key == 27:
                break

            next_frame = self._read_frame(cap)
            if next_frame is None:
                break
            frame = next_frame

    def _calibrate(
        self,
        cap: cv2.VideoCapture,
        initial_frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        print("Starting calibration...\n")
        print("Look normally at the camera")

        samples = []
        frame = initial_frame
        started_at = time.monotonic()
        last_percent = -1

        while True:
            elapsed = time.monotonic() - started_at
            progress = min(elapsed / self.calibration_seconds, 1.0)
            result = self.detector.process(frame)
            if result.ear is not None:
                samples.append(result.ear)

            display = frame.copy()
            self._draw_calibration_overlay(display, progress, result.ear is not None)
            cv2.imshow(self.window_name, display)

            percent = int(progress * 100)
            if percent != last_percent:
                filled = int(progress * 20)
                bar = "=" * filled + "-" * (20 - filled)
                print(f"\r{bar} {percent:3d}%", end="", flush=True)
                last_percent = percent

            key = cv2.waitKey(1) & 0xFF
            if not window_is_open(self.window_name) or key == 27:
                print("\nCalibration cancelled")
                return None
            if progress >= 1.0:
                break

            next_frame = self._read_frame(cap)
            if next_frame is None:
                return None
            frame = next_frame

        print()
        if len(samples) < self.calibration_min_samples:
            print(
                "Calibration failed: not enough face samples. "
                "Keep your face visible and restart."
            )
            return None

        baseline, threshold = self.detector.calibrate(samples)
        print(f"\nBaseline EAR: {baseline:.3f}")
        print(f"Personal threshold: {threshold:.3f}")
        print("\nMonitoring started")
        return frame

    def _draw_calibration_overlay(
        self,
        frame: np.ndarray,
        progress: float,
        face_detected: bool,
    ) -> None:
        height, width = frame.shape[:2]
        bar_width = min(500, width - 80)
        bar_height = 24
        x = (width - bar_width) // 2
        y = height // 2 + 30

        cv2.putText(
            frame,
            "Starting calibration...",
            (x, y - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        instruction = (
            "Look normally at the camera"
            if face_detected
            else "No face detected - face the camera"
        )
        color = (255, 255, 255) if face_detected else (0, 255, 255)
        cv2.putText(frame, instruction, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)
        fill_width = int((bar_width - 4) * progress)
        cv2.rectangle(
            frame,
            (x + 2, y + 2),
            (x + 2 + fill_width, y + bar_height - 2),
            (0, 200, 0),
            -1,
        )
        cv2.putText(
            frame,
            f"{int(progress * 100)}%",
            (x + bar_width // 2 - 25, y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def _read_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera read failed")
            return None
        return cv2.flip(frame, 1)

    def _draw_overlays(self, frame: np.ndarray, result: DrowsinessResult) -> None:
        # Eye landmarks
        if result.left_eye is not None:
            for pt in result.left_eye.astype(int):
                cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
        if result.right_eye is not None:
            for pt in result.right_eye.astype(int):
                cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)

        # EAR overlays
        if result.ear is not None:
            cv2.putText(
                frame,
                f"EAR: {result.ear:.3f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Eyes closed: {result.closed_duration_s:.2f}s",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            if self.detector.ear_threshold is not None:
                cv2.putText(
                    frame,
                    f"Personal threshold: {self.detector.ear_threshold:.3f}",
                    (30, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

        # Head pose overlays (pitch/yaw/roll)
        if result.pitch_deg is not None and result.yaw_deg is not None and result.roll_deg is not None:
            cv2.putText(
                frame,
                f"Pitch: {result.pitch_deg:+.1f}  Yaw: {result.yaw_deg:+.1f}  Roll: {result.roll_deg:+.1f}",
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Nodding detection alert
        if result.nod_detected:
            cv2.putText(
                frame,
                "NOD DETECTED!",
                (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            self.alarm.trigger()

        # Drowsiness alert (EAR-based)
        if result.is_drowsy:
            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
            self.alarm.trigger()
