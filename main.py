import numpy
import cv2
import time
import pyttsx3 # text to speech
import scipy
import mediapipe
print("all good")

engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

while True:
    print("inside while loop")
    null, frame = cap.read()
    greyScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Webcam", greyScale)
    key = cv2.waitKey(9)
    if key == 20: # press 't' to speak
        break
cap.release()
cv2.destroyAllWindows()