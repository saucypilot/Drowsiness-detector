import cv2
import numpy as np
import time
import os
import sys
import threading
import subprocess

import mediapipe as mp

WINDOW_NAME = "Drowsiness Detector"

def _window_is_open(name: str) -> bool:
    """Return True if an OpenCV HighGUI window exists and is visible."""
    try:
        prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        return prop >= 1
    except cv2.error:
        return False


# ----------------------------
# Alarm (cross-platform, non-blocking)
# ----------------------------
ALARM_MODE = "beep"   # "beep" or "wav"
ALARM_WAV_PATH = "alarm.wav"  # only used if ALARM_MODE == "wav"
ALARM_COOLDOWN = 2.0  # seconds between alarm triggers

_alarm_lock = threading.Lock()
_alarm_playing = False

def _try_run(cmd: list[str]) -> bool:
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def _play_alarm_wav():
    # WAV playback via native tools
    if sys.platform.startswith("win"):
        try:
            import winsound
            winsound.PlaySound(ALARM_WAV_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        except Exception:
            pass

    if sys.platform == "darwin":
        # macOS: afplay
        if _try_run(["afplay", ALARM_WAV_PATH]):
            return

    # Linux / other: paplay -> aplay -> ffplay
    if _try_run(["paplay", ALARM_WAV_PATH]): return
    if _try_run(["aplay", ALARM_WAV_PATH]): return
    _try_run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", ALARM_WAV_PATH])

def _play_alarm_beep():
    if sys.platform.startswith("win"):
        try:
            import winsound
            # A punchy beep pattern
            for _ in range(3):
                winsound.Beep(1200, 200)
                time.sleep(0.05)
                winsound.Beep(900, 200)
                time.sleep(0.05)
            return
        except Exception:
            pass

    # Fallback: terminal bell
    for _ in range(6):
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(0.12)

def trigger_alarm_nonblocking():
    global _alarm_playing
    with _alarm_lock:
        if _alarm_playing:
            return
        _alarm_playing = True

    def _runner():
        global _alarm_playing
        try:
            if ALARM_MODE == "wav":
                if os.path.exists(ALARM_WAV_PATH):
                    _play_alarm_wav()
                else:
                    # No wav found -> beep fallback
                    _play_alarm_beep()
            else:
                _play_alarm_beep()
        finally:
            # Let it be triggerable again shortly after
            time.sleep(0.3)
            with _alarm_lock:
                _alarm_playing = False

    threading.Thread(target=_runner, daemon=True).start()


print("all good")

# Use 0 for default webcam.
# CAP_DSHOW is Windows-only; don’t force it on other OS.
if sys.platform.startswith("win"):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit


# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# EAR helpers
def eye_aspect_ratio(pts):
    """
    pts: 6 points (np arrays) ordered:
    [left_corner, upper_left, upper_right, right_corner, lower_right, lower_left]
    """
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h  = np.linalg.norm(p1 - p4)
    return 0.0 if h == 0 else (v1 + v2) / (2.0 * h)

def lm_to_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

# MediaPipe FaceMesh landmark indices for eyes
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Drowsiness logic
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 20

closed_frames = 0
last_alarm_time = 0.0

# Grab one frame to lock in dimensions
ret0, frame0 = cap.read()
if not ret0 or frame0 is None:
    print("Camera read failed on startup")
    raise SystemExit

frame0 = cv2.flip(frame0, 1)
H, W = frame0.shape[:2]

camera_params = {
    "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or W,
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or H,
    "fps":    float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
}
print("Camera params:", camera_params)

left_eye_mask  = np.zeros((H, W), dtype=np.uint8)
right_eye_mask = np.zeros((H, W), dtype=np.uint8)
eyes_mask      = np.zeros((H, W), dtype=np.uint8)

# Enable resizing of windows
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

while True:
    if not _window_is_open(WINDOW_NAME):
        break

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if (h, w) != (H, W):
        H, W = h, w
        left_eye_mask  = np.zeros((H, W), dtype=np.uint8)
        right_eye_mask = np.zeros((H, W), dtype=np.uint8)
        eyes_mask      = np.zeros((H, W), dtype=np.uint8)

    left_eye_mask.fill(0)
    right_eye_mask.fill(0)
    eyes_mask.fill(0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear = None

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        left_pts  = np.array([lm_to_xy(lms[i], w, h) for i in LEFT_EYE])
        right_pts = np.array([lm_to_xy(lms[i], w, h) for i in RIGHT_EYE])

        ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0

        lp = left_pts.astype(np.int32)
        rp = right_pts.astype(np.int32)
        cv2.fillConvexPoly(left_eye_mask, lp, 255)
        cv2.fillConvexPoly(right_eye_mask, rp, 255)
        cv2.bitwise_or(left_eye_mask, right_eye_mask, eyes_mask)

        for p in left_pts.astype(int):
            cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)
        for p in right_pts.astype(int):
            cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)

        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames >= CONSEC_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT!", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            now = time.time()
            if (now - last_alarm_time) > ALARM_COOLDOWN:
                last_alarm_time = now
                trigger_alarm_nonblocking()

    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.3f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"ClosedFrames: {closed_frames}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if not _window_is_open(WINDOW_NAME):
        break
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
