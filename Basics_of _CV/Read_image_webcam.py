import cv2
import numpy as np

#img = cv2.imread("")
frameWidth = 700
frameHeight = 700

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    success, img = cap.read()

    cv2.imshow("Image", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
      break
