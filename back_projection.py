'''

Histogram Backprojection:

In this blog, we will discuss Histogram Backprojection, a technique that is used for image segmentation or finding objects of interest in an image.
It was proposed by Michael J. Swain, Dana H. Ballard in their paper Indexing via color histograms, Third international conference on computer vision,1990.
This was one of the first works that use color to address the classic problem of Classification and Localisation in computer vision.

To understand this technique, knowledge of histograms (particularly 2-D histograms) is a must. 
If you haven’t encountered 2-d histograms yet, I suggest you to read What is a 2-D histogram?     '''



# importing libraies
import cv2
import numpy as np
import matplotlib.pyplot as plt

# resize the image
height_Img = 700
width_Img = 500

# importing Images
image = cv2.imread("C:\\Users\\Rafiun Nesha\\Desktop\\images\\goalkeeper.jpg")
main_image = cv2.resize(image, (height_Img, width_Img))
main_img_hsv = cv2.cvtColor(main_image, cv2.COLOR_BGR2HSV)

roi_img = cv2.imread("C:\\Users\\Rafiun Nesha\\Desktop\\images\\pitch_ground.jpg")
roi = cv2.resize(roi_img, (height_Img, width_Img))
roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

hue, saturation, value = cv2.split(roi_hsv)

# Draw a histogram of roi
roi_hist = cv2.calcHist([roi_hsv],[0,1], None, [180,256], [0,180,0,256])
mask = cv2.calcBackProject([main_img_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

# add filter for removing noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

# applying bitwise for background
mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_and(main_image, mask)

cv2.imshow("Mask", mask)
cv2.imshow("Main_image", main_image)
cv2.imshow("Result", result)
cv2.imshow("ROI", roi)
cv2.imshow("ROI_hsv", roi_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
