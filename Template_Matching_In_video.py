import cv2
import numpy

cap = cv2.VideoCapture(0)
template = cv2.imread("C:\\Users\\Rafiun Nesha\\Desktop\\images\\Template_pen.jpg", cv2.IMREAD_GRAYSCALE)

w, h = template.shape[::-1]

while True:
    _,frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    loc = numpy.where(result >= 0.6)

    for pt in zip(*loc[::-1]):
     cv2.rectangle(frame , pt, (pt[0] + w, pt[1] + h ), (0,255,0),3)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
     break

cap.release()
cv2.destroyAllWindows()

