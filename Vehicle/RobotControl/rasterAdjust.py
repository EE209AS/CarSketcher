import cv2
import numpy as np

class rasterAdjust:

    def __init__(self):
        pass

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

        if cv2.__version__ == '3.1.0':
            img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        else :
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
            # cv2.drawContours(frame, [contourMax], 0, (0, 255, 0), 5)
            # cv2.circle(frame,(cXMax,cYMax),5,(0,0,255),-1)
        return (frame, cXMax, cYMax, areaMax, contourMax)




    def RasterAdjust(self, frame):
        (frameTrack, cx, cy, areaMax, contour)=self.trackPaper(frame)
        if len(contour) == 0:
            return []
        x=[]
        y=[]
        i=0
        while(i<len(contour)):
            x.append(contour[i][0][0])
            y.append(contour[i][0][1])
            i=i+1

        # x.sort()
        # y.sort()

        nx=np.array(x)
        ny=np.array(y)
        idmin=np.argmin(ny)
        mx, my = nx[idmin], ny[idmin]

        ycenmin=1000000
        xcenmin=0
        for p in contour:
            if(abs(p[0][0]-cx)<50):
                if ycenmin>p[0][1]:
                    ycenmin=p[0][1]
                    xcenmin=p[0][0]

        # print cx, ycenmin, i
        # cv2.circle(frameTrack,(xcenmin,ycenmin),5,(0,0,255),-1)

        if mx > xcenmin:
            r = float(ycenmin-my)/(xcenmin-mx)
        else :
            r = float(my - ycenmin) / (mx - xcenmin)
        ctr = -1
        print r
        if r > 0.2:
            ctr = 1
            # cv2.putText(frameTrack,"right",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
        elif r < -0.2:
            ctr = -1
            # cv2.putText(frameTrack,"left",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
        else:
            ctr = 0
            # cv2.putText(frameTrack,"go",(cx-150,cy-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)


        return (ctr, frameTrack)



