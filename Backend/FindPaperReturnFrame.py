import cv2
import numpy as np

class FindPaperReturnFrame:

    lowW=[]
    highW=[]
    width=[]
    height=[]
    VisionArea=[]

    def __init__(self, width, height):
        self.lowW=int(float(1)/3*width)
        self.highW=int(float(2)/3*width)
        self.VisionArea=width*height
        self.width=width
        self.height=height

    def detectRectangle(self, c):
            shape=False
            peri=cv2.arcLength(c,True)
            approx=cv2.approxPolyDP(c,0.04*peri,True)

            if len(approx)==4:
                (x,y,w,h)=cv2.boundingRect(approx)
                shape = True

            return shape

    def trackPaper(self, frame):

        resized = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)
        ratio=frame.shape[0]/float(resized.shape[0])

        gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(5,5),0)

        cv2.normalize(blurred,blurred,0,255,cv2.NORM_MINMAX)
        thresh=cv2.threshold(blurred,150,255,cv2.THRESH_BINARY)[1]

        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        areaMax=0
        cXMax=-1
        cYMax=-1
        cX=-1
        cY=-1
        contourMax=[]
        for c in contours:
                shape = self.detectRectangle(c)
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

        if areaMax != 0:
            contourMax = contourMax*ratio
            contourMax=np.int0(contourMax)
            cv2.drawContours(frame, [contourMax], 0, (0, 255, 0), 5)
            cv2.circle(frame,(cXMax,cYMax),5,(0,0,255),-1)
        return (frame, cXMax, cYMax, areaMax)

    def ReadCameraDataReturnJPG(self, frame):

        # get the 4 corners of the paper
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        FourCorners = cv2.goodFeaturesToTrack(gray,4,0.01,10)
        FourCorners = np.int0(FourCorners)

        # get the frame to show
        (frameTrack, cx, cy, areaMax)=self.trackPaper(frame)
        cv2.rectangle(frameTrack,(self.lowW,0),(self.highW,int(self.height)),(0,0,255),5)
        ctr=-1
        if cx != -1:
            if cx>=self.lowW and cx<= self.highW:
                cv2.putText(frameTrack,"Go Straight",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
                ctr=0
            elif cx<self.lowW:
                cv2.putText(frameTrack,"Turn Left",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
                ctr=1
            else:
                cv2.putText(frameTrack,"Turn Right",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
                ctr=2

            if float(areaMax)/self.VisionArea > 0.1:
                cv2.putText(frameTrack,"Stop",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
                ctr=3

        for i in FourCorners:
            x,y = i.ravel()
            cv2.circle(frameTrack, (x,y), 15, (0,0,255), -1)

        # encode to jpg
        retval, buf=cv2.imencode('.jpg',frameTrack)

        # buf is the jpg to show and ctr is the control predicted

        return (buf, ctr)
