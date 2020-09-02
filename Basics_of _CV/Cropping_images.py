import cv2
import numpy as np

path = cv2.imread("download (5).jpg")
print(path.shape)

Width, Height = 500, 500
imgResize = cv2.resize(path,(Width, Height))
print(imgResize.shape)

imgCropped = path[300:300, 400:300]
imgCropResize = cv2.resize(imgResize,(path.shape[1], path.shape[0]))

cv2.imshow("img", imgResize)
cv2.imshow("image", path)
cv2.imshow("crop_img", imgCropResize)

cv2.waitKey(0)