import os
import subprocess
import sys
import threading
import time
from typing import Sequence


class AlarmPlayer:
    """Play drowsiness alarms asynchronously with cooldown control."""

    def __init__(self, mode: str = "beep", wav_path: str = "alarm.wav", cooldown: float = 2.0) -> None:
        self.mode = mode
        self.wav_path = wav_path
        self.cooldown = cooldown
        self._lock = threading.Lock()
        self._playing = False
        self._last_trigger = 0.0

    def trigger(self) -> None:
        """Start alarm playback if it is not already running or cooling down."""
        now = time.time()
        with self._lock:
            if self._playing or (now - self._last_trigger) < self.cooldown:
                return
            self._playing = True
            self._last_trigger = now

        threading.Thread(target=self._run_alarm, daemon=True).start()

    def _run_alarm(self) -> None:
        try:
            if self.mode == "wav" and os.path.exists(self.wav_path):
                self._play_alarm_wav()
            else:
                self._play_alarm_beep()
        finally:
            time.sleep(0.3)
            with self._lock:
                self._playing = False

    @staticmethod
    def _try_run(cmd: Sequence[str]) -> bool:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _play_alarm_wav(self) -> None:
        if sys.platform.startswith("win"):
            try:
                import winsound

                winsound.PlaySound(self.wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass

        if sys.platform == "darwin" and self._try_run(["afplay", self.wav_path]):
            return

        if self._try_run(["paplay", self.wav_path]):
            return
        if self._try_run(["aplay", self.wav_path]):
            return
        self._try_run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.wav_path])

    def _play_alarm_beep(self) -> None:
        if sys.platform.startswith("win"):
            try:
                import winsound

                for _ in range(3):
                    winsound.Beep(1200, 200)
                    time.sleep(0.05)
                    winsound.Beep(900, 200)
                    time.sleep(0.05)
                return
            except Exception:
                pass

        for _ in range(6):
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.12)
