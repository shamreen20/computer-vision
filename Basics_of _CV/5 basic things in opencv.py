import cv2
import numpy as np

img = cv2.imread("download (2).jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img_gray,(5,5),0)
img_edge = cv2.Canny(imgBlur,150,200)
img_dilation = cv2.dilate(img_edge, (5,5), iterations=10)
img_Erode = cv2.erode(img_edge,(5,5), iterations=0)



cv2.imshow("gray_name", img_gray)
cv2.imshow("image",img)
cv2.imshow("imgBlur",imgBlur)
cv2.imshow("img_edge", img_edge)
cv2.imshow("img_dilation", img_dilation)
cv2.imshow("img_Erode", img_Erode)


cv2.waitKey(0)