import cv2
import numpy as np

cap = cv2.VideoCapture(0)

width=cap.get(3)
height=cap.get(4)
lowW=int(float(1)/3*width)
highW=int(float(2)/3*width)
lowH=int(float(1)/3*height)
highH=int(float(2)/3*height)

print width,height,lowW,highW,lowH,highH

ret, img = cap.read()
edges = cv2.Canny(img,100,200)

cap.release()
cv2.destroyAllWindows()

