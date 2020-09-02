import cv2
import numpy as np

img = cv2.imread("squares.jpg")
print(img.shape)

width, height = 500,500
pts1 = np.float32([[80,100],[200, 40],[50,500],[500,500]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix,(width,height))

for x in range(0,4):
    cv2.circle(img,(pts1[x][0], pts2[x][1]), 15, (0,255,0), cv2.FILLED)

cv2.imshow("Image", img)
cv2.imshow("ImgOutput", imgOutput)
cv2.waitKey(0)
