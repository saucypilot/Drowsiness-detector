import cv2
import numpy as np
import time

import mediapipe as mp

# Optional: text-to-speech alert
USE_TTS = True
if USE_TTS:
    import pyttsx3
    engine = pyttsx3.init()

print("all good")

# --------- Camera ----------
# Use 0 for default webcam. On Windows, CAP_DSHOW often behaves better.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit

# --------- MediaPipe FaceMesh ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------- EAR helpers ----------
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

# MediaPipe FaceMesh landmark indices for eyes (good baseline for EAR)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --------- Drowsiness logic ----------
EAR_THRESHOLD = 0.22     # tune 0.18 - 0.25 depending on your face/camera
CONSEC_FRAMES = 20       # how many consecutive frames EAR must be low
ALARM_COOLDOWN = 2.0     # seconds between voice alerts

closed_frames = 0
last_alarm_time = 0.0

while True:
    print("inside while loop")

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # MediaPipe expects RGB (NOT grayscale)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear = None

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark

        left_pts  = np.array([lm_to_xy(lms[i], w, h) for i in LEFT_EYE])
        right_pts = np.array([lm_to_xy(lms[i], w, h) for i in RIGHT_EYE])

        ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0

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
            if USE_TTS and (now - last_alarm_time) > ALARM_COOLDOWN:
                last_alarm_time = now
                engine.say("Wake up")
                engine.runAndWait()

    # HUD text
    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.3f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"ClosedFrames: {closed_frames}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
