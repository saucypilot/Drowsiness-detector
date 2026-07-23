# Webcam Drowsiness Detection (EAR + MediaPipe)

This project is a real-time drowsiness detection system using your webcam. It
tracks facial landmarks, measures eye openness with Eye Aspect Ratio (EAR),
estimates head nodding, and combines those signals into a single drowsiness risk
score.

---

## What This Program Does

At a high level, the app:

* Captures live video from your webcam
* Detects one face using MediaPipe
* Extracts eye landmarks
* Computes Eye Aspect Ratio (EAR) every frame
* Calibrates a personal open-eye baseline at startup
* Tracks eye closure duration, blink behavior, PERCLOS, and head nods
* Combines those signals into a 0-100 drowsiness risk score
* Triggers the alarm when the score reaches the drowsy range

The driver-state panel shows:

```text
DRIVER STATE

Drowsiness Risk: 72%
[risk bar]

Eyes:    Frequent closure
PERCLOS: 31%
Head:    2 nods detected
Blink:   18/min

STATUS: DROWSY
```

---

## Why EAR Works

EAR measures the ratio of eye height to eye width.

When your eyes are open:

* Vertical distance between eyelids is large
* EAR is relatively high

When your eyes close:

* Vertical distance collapses
* EAR drops sharply

The formula:

```text
EAR = (vertical_1 + vertical_2) / (2 * horizontal)
```

EAR is useful, but it is not enough by itself. Different people, glasses,
camera position, and camera distance can shift the absolute value. This app
therefore calibrates EAR per session and uses EAR as one input into a combined
risk score.

---

## Technologies Used

* Python 3
* OpenCV - webcam capture and drawing
* MediaPipe - facial landmarks
* NumPy - vector math
* pyttsx3 or system beep support for optional alarms

No cloud model is used while the app runs.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe numpy pyttsx3
```

If you do not want voice alerts, you can skip `pyttsx3`.

---

## Running the Program

```bash
python main.py
```

* Press ESC to exit.
* Make sure your face is visible and well lit.
* Webcam index defaults to `0`.

On startup, the app calibrates for five seconds. Look normally at the camera
with your eyes open until the progress bar completes.

---

## Configuration

The app calibrates EAR automatically for each person during startup. The median
observed EAR becomes the baseline, and the personal threshold is 75% of that
baseline:

```python
baseline_ear = median(calibration_samples)
ear_threshold = baseline_ear * 0.75
```

The relevant defaults are:

```python
CALIBRATION_SECONDS = 5.0
THRESHOLD_RATIO = 0.75
DROWSY_DURATION_SECONDS = 1.5
RISK_WINDOW_SECONDS = 60.0
LONG_BLINK_MIN_SECONDS = 0.7
ALARM_COOLDOWN = 2.0
USE_TTS = True
```

What they mean:

* `CALIBRATION_SECONDS`: how long to observe normal, open eyes at startup
* `THRESHOLD_RATIO`: personal threshold as a fraction of median baseline EAR
* `DROWSY_DURATION_SECONDS`: continuous eye closure duration that contributes full eye-closure risk
* `RISK_WINDOW_SECONDS`: rolling window used for PERCLOS, blink rate, long blinks, and nod count
* `LONG_BLINK_MIN_SECONDS`: minimum closure duration counted as a long blink
* `ALARM_COOLDOWN`: minimum seconds between alarm sounds

Glasses, camera position, distance, and natural eye shape are accounted for by
per-session calibration. An explicit `ear_threshold` can still be passed to
`DrowsinessDetectorApp` to bypass calibration when needed.

---

## Drowsiness Logic

The detector now uses a combined drowsiness risk score instead of independent
EAR and nod alarms.

Startup calibration:

* Collect five seconds of normal open-eye EAR samples
* Use the median sample as the baseline
* Set the personal threshold to `baseline * 0.75`

Runtime tracking:

* EAR below threshold starts or continues the eye-closure timer
* EAR above threshold records a blink event and resets the closure timer
* PERCLOS is computed from elapsed closed time in a rolling 60-second window
* Long blinks are closures of at least 0.7 seconds
* Nod events are counted in the same rolling window

Risk score:

```python
risk_score = (
    prolonged_eye_closure * 0.45
    + frequent_long_blinks * 0.20
    + nod_score * 0.20
    + perclos_score * 0.15
)
```

If continuous eye closure reaches `DROWSY_DURATION_SECONDS`, the score is
floored into the drowsy band even if the rolling-window metrics are still
warming up.

Classification:

* `0-30`: ALERT
* `31-60`: FATIGUE WARNING
* `61-80`: DROWSY
* `81-100`: CRITICAL

The alarm triggers from the combined score, not from separate EAR and nod
checks.

---

## Code Structure

* `main.py`: app entry point
* `app.py`: webcam loop, calibration UI, driver-state overlay, alarm trigger
* `detector.py`: MediaPipe landmarks, EAR, rolling risk state, risk scoring
* `head_pose.py`: pitch/yaw/roll estimation and nod detection
* `alarm.py`: asynchronous alarm playback with cooldown

---

## Performance Notes

* Frame resolution changes are handled dynamically
* Risk metrics use elapsed time, so behavior is independent of camera FPS
* Real-time performance is typically 30-60 FPS on a normal laptop
* Printing every frame will reduce FPS noticeably

---

## Limitations

This is not medical software.

It can fail if:

* You look down for long periods
* Lighting is poor
* Camera angle is extreme
* Face is partially occluded
* Glasses cause glare

It estimates drowsiness-related visual signals, not cognitive fatigue.

---

## Use Cases

* Studying late at night
* Long coding sessions
* Prototype for driver drowsiness systems
* Computer vision learning project
* MediaPipe / OpenCV practice
