import cv2
import numpy as np

class CarCameraControl:

    width=[]
    height=[]
    cap=[]

    def __init__(self):
        pass


    # Camera Control
    def InitCamera(self):
        self.cap = cv2.VideoCapture(1)
        self.width=self.cap.get(3)
        self.height=self.cap.get(4)

    def DestoryCamera(self):
        self.cap.release()

    def GetOneVideoFrame(self):
        ret, frame = self.cap.read()
        return frame

    def CaptureThePicToDraw(self, oriframe):
        edge = cv2.Canny(oriframe,100,200)
        # resize to 64 * 48
        edge = cv2.resize(edge, (0,0), fx=0.1, fy=0.1)
        return edge

    def Get4Corners(self, oriframe):
        gray = cv2.cvtColor(oriframe, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray,4,0.01,10)
        corners = np.int0(corners)
        return corners

    def ConvertFrameToJPG(self, frame):
        retval, buf=cv2.imencode('.jpg',frame)
        retrun buf



    # # Movement Control
    # def goStraight(self):
    #     i=0
    #
    # def goLeft(self):
    #     i=0
    #
    # def goRight(self):
    #     i=0
    #
    # def goBack(self):
    #     i=0
    #
    # def stop(self):
    #     i=0