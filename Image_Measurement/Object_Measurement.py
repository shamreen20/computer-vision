import cv2
import numpy as np
import utils


webcam = False
path = 'cards.jpg'
ImageWidth = 500
ImageHeight = 500
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale = 3
wp = 210 * scale
hp = 297 * scale

while True:
   if webcam: success , img = cap.read()
   else: img = cv2.imread(path)
   img, conts= utils.getContours(img, showCanny=True, minArea = 50000, filter=4)

   if len(conts)!= 0:
      biggest = conts[0][2]
      imgWrap = utils.warpImg(img, biggest, wp, hp)
      imgContours2, conts2 = utils.getContours(imgWrap, minArea=2000, filter = 4, cThr=[50,50], draw = False)

      if len(conts)!= 0:
         for obj in conts2:
            cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
            nPoints = utils.reorder(obj[2])
            nW = round((utils.findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10), 1)
            nH = round((utils.findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10),1)
            cv2.arrowedLine(imgContours2,(nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255,0,255), 3,8,0,0.05)
            x, y, w, h = obj[3]
            cv2.putText(imgContours2,'{}cm'.format(nW), (x+30, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,(255,0,255), 2)
            cv2.putText(imgContours2, '{}cm'.format(nH), (x-70,y+h//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,(255,0,255), 2)
      cv2.imshow('A4', imgContours2)
   img = cv2.resize(img,(0,0), None, 0.5,0.5)
   cv2.imshow('original', img)

   cv2.waitKey(1)