import cv2
import numpy as np
import time
import threading

import mediapipe as mp

# Alarm sound when drowsy.
USE_ALARM = True

# Cooldown between alarm triggers (seconds)
ALARM_COOLDOWN = 2.0

# Alarm mode:
#   - "wav": plays a .wav file asynchronously (Windows only via winsound)
#   - "beep": uses a short beep pattern (best "no extra files" option)
ALARM_MODE = "beep"
ALARM_WAV_PATH = "alarm.wav"  # used only if ALARM_MODE == "wav"

# Beep pattern (frequency Hz, duration ms) used when ALARM_MODE == "beep"
BEEP_PATTERN = [(1200, 200), (900, 200), (1200, 200), (900, 400)]

# Windows sound backend
try:
    import winsound  # Windows only
except Exception:
    winsound = None

_alarm_lock = threading.Lock()
_alarm_playing = False

def _play_alarm_blocking():
    """Runs in a background thread so the camera loop doesn't freeze."""
    global _alarm_playing
    try:
        if ALARM_MODE == "wav" and winsound:
            # Asynchronous play; we still hold a short window as "playing"
            winsound.PlaySound(ALARM_WAV_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            time.sleep(1.2)
        elif winsound:
            for freq, dur in BEEP_PATTERN:
                winsound.Beep(int(freq), int(dur))  # blocks, hence the thread
        else:
            # Cross-platform fallback: terminal bell (may be muted depending on OS/settings)
            for _ in range(3):
                print("\a", end="", flush=True)
                time.sleep(0.25)
    finally:
        with _alarm_lock:
            _alarm_playing = False

def trigger_alarm():
    """Start alarm if not already playing."""
    global _alarm_playing
    if not USE_ALARM:
        return
    with _alarm_lock:
        if _alarm_playing:
            return
        _alarm_playing = True
    t = threading.Thread(target=_play_alarm_blocking, daemon=True)
    t.start()

print("all good")

# Use 0 for default webcam. On Windows, CAP_DSHOW often behaves better.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)  # normalized to pixel coords

# MediaPipe FaceMesh landmark indices for eyes (good baseline for EAR)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Drowsiness logic
EAR_THRESHOLD = 0.22     # tune 0.18 - 0.25 depending on your face/camera
CONSEC_FRAMES = 20       # how many consecutive frames EAR must be low

closed_frames = 0
last_alarm_time = 0.0

# Camera parameters + preallocated eye masks
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

cv2.namedWindow("Drowsiness Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drowsiness Detector", 960, 540)

while True:
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

        # Draw eye points
        for p in left_pts.astype(int):
            cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)
        for p in right_pts.astype(int):
            cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)

        # Drowsiness counter
        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0

        # Trigger alert
        if closed_frames >= CONSEC_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT!", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            now = time.time()
            if (now - last_alarm_time) > ALARM_COOLDOWN:
                last_alarm_time = now
                trigger_alarm()

    # HUD text
    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.3f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"ClosedFrames: {closed_frames}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
