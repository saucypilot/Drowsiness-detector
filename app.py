import sys
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
    ) -> None:
        self.camera_index = camera_index
        self.window_name = window_name
        self.window_size = window_size
        self.alarm = AlarmPlayer(mode=alarm_mode, wav_path=alarm_wav_path, cooldown=alarm_cooldown)
        self.detector = DrowsinessDetector()

    def run(self) -> None:
        print("all good")
        cap = self._open_camera()
        try:
            frame0 = self._read_initial_frame(cap)
            h0, w0 = frame0.shape[:2]
            self._log_camera_params(cap, w0, h0)
            self._setup_window()
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

    def _read_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera read failed")
            return None
        return cv2.flip(frame, 1)

    def _draw_overlays(self, frame: np.ndarray, result: DrowsinessResult) -> None:
        if result.left_eye is not None:
            for pt in result.left_eye.astype(int):
                cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
        if result.right_eye is not None:
            for pt in result.right_eye.astype(int):
                cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)

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
                f"ClosedFrames: {result.closed_frames}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
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
