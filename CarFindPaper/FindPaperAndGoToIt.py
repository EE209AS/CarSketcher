import cv2
import numpy as np
import time

def detectRectangle(c):
        shape=False
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.04*peri,True)

        if len(approx)==4:
            (x,y,w,h)=cv2.boundingRect(approx)
            shape = True

        return shape

def trackPaper(frame):

    resized = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)
    ratio=frame.shape[0]/float(resized.shape[0])

    gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(5,5),0)

    cv2.imshow("ad",resized)

    cv2.normalize(blurred,blurred,0,255,cv2.NORM_MINMAX)
    thresh=cv2.threshold(blurred,150,255,cv2.THRESH_BINARY)[1]

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    areaMax=0
    cXMax=-1
    cYMax=-1
    contourMax=[]
    for c in contours:
            shape = detectRectangle(c)
            if shape == True:
                c = np.float32(c)
                M = cv2.moments(c)
                if M["m00"]!=0:
                    cX = int((M["m10"]/M["m00"]) * ratio)
                    cY = int((M["m01"]/M["m00"]) * ratio)
                else:
                    cX=-1
                    cY=-1

                if cX != -1:
                    area = cv2.contourArea(c)
                    if area>areaMax:
                        areaMax=area
                        contourMax=c
                        cXMax=cX
                        cYMax=cY
            else:
                cX=-1
                cY=-1

    # if areaMax != 0:
        # contourMax = contourMax*ratio
        # contourMax=np.int0(contourMax)
        # cv2.drawContours(frame, [contourMax], 0, (0, 255, 0), 5)
        # cv2.circle(frame,(cXMax,cYMax),5,(0,0,255),-1)
    return (frame, cXMax, cYMax)

#################################################

cap = cv2.VideoCapture(0)

width=cap.get(3)
height=cap.get(4)
lowW=int(float(1)/3*width)
highW=int(float(2)/3*width)


count=0
while(True):

    keypressed=cv2.waitKey(1) & 0xFF
    if keypressed == ord('q'):
        break

    time.sleep(1)

    ret, frame = cap.read()

    (frameTrack, cx, cy)=trackPaper(frame)

    # cv2.rectangle(frameTrack,(lowW,0),(highW,int(height)),(0,0,255),5)

    if cx != -1:
        if cx>lowW and cx< highW:
            # cv2.putText(frameTrack,"Correct",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
            print "Correct"
        elif cx<lowW:
            # cv2.putText(frameTrack,"Turn Left",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
            print "Left"
        else:
            # cv2.putText(frameTrack,"Turn Right",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
            print "Right"

    cv2.imshow('frameTrack',frameTrack)



cap.release()
cv2.destroyAllWindows()