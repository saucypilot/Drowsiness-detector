# Webcam Drowsiness Detection (Eye Aspect Ratio + MediaPipe)

This project is a **real-time drowsiness detection system** using your webcam. It tracks facial landmarks, measures how open your eyes are using **Eye Aspect Ratio (EAR)**, and triggers a visual (and optional voice) alert if your eyes remain closed for too long.
---

## What This Program Does

At a high level, the script:

* Captures live video from your webcam
* Detects a face using **MediaPipe FaceMesh**
* Extracts eye landmarks
* Computes **Eye Aspect Ratio (EAR)** every frame
* If EAR stays below a threshold for *N consecutive frames*:

  * Displays a **DROWSINESS ALERT**
  * Optionally speaks “Wake up” using text-to-speech

Think of it like a guard watching a door:

* Door open → you’re awake
* Door briefly closes → blink (ignored)
* Door stays shut → alarm

---

## Why EAR Works (Conceptual Explanation)

EAR measures the **ratio of eye height to eye width**.

When your eyes are open:

* Vertical distance between eyelids is large
* EAR is relatively high

When your eyes close:

* Vertical distance collapses
* EAR drops sharply

The math is simple but effective:

```
EAR = (vertical_1 + vertical_2) / (2 * horizontal)
```

This stays surprisingly stable across different people and lighting conditions.

---

## Technologies Used

* **Python 3**
* **OpenCV (cv2)** – webcam capture + drawing
* **MediaPipe FaceMesh** – 468 facial landmarks
* **NumPy** – vector math
* **pyttsx3 (optional)** – offline text-to-speech

No cloud models. No internet required.

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

> If you don’t want voice alerts, you can skip `pyttsx3`.

---

## Running the Program

```bash
python main.py
```

* Press **ESC** to exit.
* Make sure your face is visible and well-lit.
* Webcam index defaults to `0`.

---

## Configuration (Important)

These constants control behavior:

```python
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 20
ALARM_COOLDOWN = 2.0
USE_TTS = True
```

### What they mean

* **EAR_THRESHOLD**

  * Lower = stricter eye closure detection
  * Typical range: `0.20 – 0.25`

* **CONSEC_FRAMES**

  * Number of frames eyes must stay closed
  * At ~30 FPS, 20 frames ≈ 0.66 seconds

* **ALARM_COOLDOWN**

  * Minimum seconds between voice alerts

* **USE_TTS**

  * Toggle voice alerts on/off

If you wear glasses or have a narrow eye shape, you *will* need to tune `EAR_THRESHOLD`.

---

## How the Code Is Structured

### Webcam Initialization

Uses OpenCV to capture live frames. The frame is mirrored so movement feels natural.

### Face Detection

MediaPipe FaceMesh tracks facial landmarks in real time. Only one face is processed for performance and simplicity.

### Eye Landmark Selection

Each eye uses **6 specific landmarks** chosen for stable EAR computation.

### EAR Calculation

Distances between eyelid landmarks are computed using Euclidean distance.

### Drowsiness Logic

* EAR below threshold → increment counter
* EAR above threshold → reset counter
* Counter exceeds limit → alert

This prevents false positives from blinking.

---

## Performance Notes

* Masks are pre-allocated to reduce memory churn
* Frame resolution changes are handled dynamically
* Real-time performance is typically 30–60 FPS on a normal laptop

⚠️ Remove debug prints inside the loop if FPS drops — printing every frame will tank performance.

---

## Limitations (Read This)

This is **not medical software**.

It can fail if:

* You look down for long periods
* Lighting is poor
* Camera angle is extreme
* Face is partially occluded
* Glasses cause glare

It detects **eye closure**, not cognitive fatigue.

---

## Use Cases

* Studying late at night
* Long coding sessions
* Prototype for driver drowsiness systems
* Computer vision learning project
* MediaPipe / OpenCV practice

