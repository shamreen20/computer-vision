import cv2
import numpy as np

from PIL import ImageGrab

def CaptureScreen(bbox=(50,50,690,530)):
    capscr = np.array(ImageGrab.grab(bbox))
    capscr = cv2.cvtColor(capscr, cv2.COLOR_BGR2RGB)
    return capscr

while True:
    timer = cv2.getTickCount()
    img = CaptureScreen()
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    cv2.putText(img, 'FPS {}'.format(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 230, 20), 2)
    cv2.imshow('Screen Capture', img)
    cv2.waitKey(0)