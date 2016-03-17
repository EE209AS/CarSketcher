import cv2
import numpy as np

class FindPaperCornerOnJpg:

    # lowW=[]
    # highW=[]
    width=None
    height=None
    # VisionArea=[]

    # def __init__(self, width, height):
    #     self.lowW=int(float(1)/3*width)
    #     self.highW=int(float(2)/3*width)
    #     self.VisionArea=width*height
    #     self.width=width
    #     self.height=height
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
        return (frame, cXMax, cYMax, areaMax, contourMax)
    def moving_average(self, a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def FindCornersFromContour(self, filename, draw=True):
        frame=cv2.imread(filename)
        self.width, self.height = frame.shape[1], frame.shape[0]
        # get the frame to show
        (frameTrack, cx, cy, areaMax, contour)=self.trackPaper(frame)
        if len(contour) == 0:
            return []
        tan = []
        i = 0
        while (i < len(contour)):
            if i == len(contour) - 1:
                r = 0
            else :
                r = i + 1
            if contour[i][0][1] - contour[r][0][1] == 0:
                val = 10000
            else :
                val = (contour[i][0][0] - contour[r][0][0]) / (contour[i][0][1] - contour[r][0][1])
            tan.append(val)
            i += 1
        # filtering!
        tan = self.moving_average(tan, 5)
        print tan
        # peak detection
        tan[0] = np.absolute(tan[0] - tan[len(tan) - 1])
        for i in range(1, len(tan)):
            tan[i] = np.absolute(tan[i] - tan[i - 1])

        threshold = np.mean(tan)
        peaks = []
        for i in range(0, len(tan)):
            l = i - 1
            r = i + 1
            if i == 0:
                l = len(tan) - 1
            if i == len(tan) - 1:
                r = 0
            if tan[i] > tan[l] and tan[i] > tan[r] and tan[i] > threshold:
                # peak detected!
                peaks.append(i)
        print peaks
        if len(peaks) < 4:
            raise NameError('not enough peaks')
        else:
            print 'got a lot: ', len(peaks)
            cand = []
            for i in peaks:
                cand.append(tan[i])
            cand.sort()
            res = []
            for i in peaks:
                if tan[i] >= cand[3]:
                    res.append(i)
        # use patches
        print res
        patch_size = 10
        cs = []
        for i in res:
            x0, y0 = contour[i][0][0], contour[i][0][1]
            ROI = frame[y0-patch_size:y0+patch_size, x0-patch_size:x0+patch_size]
            gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            corner = cv2.goodFeaturesToTrack(gray,1,0.01,10)
            for j in corner:
                cx,cy = j.ravel()
                cx=cx+x0 - patch_size
                cy=cy+y0 - patch_size
                cs.append([cx,cy])
                if draw:
                    cv2.circle(frame, (int(cx),int(cy)), 15, (0,0,255), -1)
        if draw:
            cv2.imwrite(filename + 'corners.jpg', frame)
        return cs

    def FindCorners(self, filename, draw=True):

        frame=cv2.imread(filename)
        self.width, self.height = frame.shape[1], frame.shape[0]
        # get the frame to show
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

        x.sort()
        y.sort()

        # cv2.circle(frame,(x[0],y[0]),10,(0,0,255),-1)
        # cv2.circle(frame,(x[len(x)-1],y[len(x)-1]),10,(0,0,255),-1)
        # print x[0], x[len(x)-1], y[0], y[len(y)-1]
        bolder = 30
        if x[0] > bolder:
            x[0]=x[0]-bolder
        if y[0] > bolder:
            y[0]=y[0]-bolder
        if self.width-x[len(x)-1]>bolder:
            x[len(x)-1]=x[len(x)-1]+bolder
        if self.height-y[len(y)-1]>bolder:
            y[len(y)-1]=y[len(y)-1]+bolder

        # cv2.rectangle(frameTrack,(x[0],y[0]),(x[len(x)-1],y[len(y)-1]),(0,0,255),5)
        # cv2.imwrite(filename + 'bullshit.jpg', frameTrack)

        ROI = frame[y[0]:y[len(y)-1], x[0]:x[len(x)-1]]
        # get the 4 corners of the paper
        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        # thresh=cv2.threshold(gray,50,255,cv2.THRESH_BINARY)[1]

        FourCorners = cv2.goodFeaturesToTrack(gray,4,0.01,10)
        FourCorners = np.int0(FourCorners)
        cs = []
        for i in FourCorners:
            cx,cy = i.ravel()
            cx=cx+x[0]
            cy=cy+y[0]
            cs.append([cx,cy])
            if draw:
                cv2.circle(frame, (cx,cy), 15, (0,0,255), -1)
        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 5)
        if draw:
            cv2.imwrite(filename + 'corners.jpg', frame)
        return cs
        # return frame
    def Dist(self, x1, x2):

        if isinstance(x1, np.ndarray):
            return np.linalg.norm(x1 - x2)

        return np.linalg.norm(np.array(x1) - np.array(x2));

    def hasPaper(self, filename, draw=False):

        cs = self.FindCornersFromContour(filename, draw)
        if not cs:
            return None

        edges = []
        for i in range(0,4):
            for j in range(i + 1,4):
                edges.append(self.Dist(cs[i], cs[j]))

        edges.sort()
        r_std1 = np.abs(edges[0] - edges[1]) / float(edges[0] + edges[1])
        r_std2 = np.abs(edges[2] - edges[3]) / float(edges[2] + edges[3])
        r_std3 = np.abs(edges[4] - edges[5]) / float(edges[4] + edges[5])
        # detection success!
        # if np.std([r_std1, r_std2, r_std3]) < 0.1:
        if np.std([r_std1, r_std2]) < 0.1:
            return cs
        else:
            print 'fk ', np.std([r_std1, r_std2, r_std3])
            return None
        return cs
