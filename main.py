import cv2
import os

print("cv2.data.haarcascades =", cv2.data.haarcascades)

cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
print("Full path:", cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Failed to load cascade from {cascade_path}")

cap = cv2.VideoCapture(0) 

if cap.isOpened():
    print("Webcam opened successfully.")
else:
    print("Error: Could not open webcam.")
    exit()