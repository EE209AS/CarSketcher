import cv2
import numpy as np
import subprocess as sp
from rasterAdjust import rasterAdjust
from CarCameraControl import CarCameraControl



def rasterCtr():
    cc=CarCameraControl()
    cc.InitCamera()
    ra=rasterAdjust()

    frame = cc.GetOneVideoFrame()
    cc.DestoryCamera()
    ctr=-2
    cx=0
    cy=0
    try:
        ctr, frame, cx, cy = ra.RasterAdjust(frame)
    except:
        print "error"


    print "==========", cx, cy

    if ctr == 1:
        sp.Popen(['./right.out']).wait()
        print "right"
    elif ctr == -1:
        sp.Popen(['./left.out']).wait()
        print "left"
    elif ctr == 0:
        print "go"
    else:
        print "no adjust"





img = np.zeros((6,6,3), np.uint8)
cv2.rectangle(img,(1, 1),(4, 4),(255,255,255),1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th = gray / 255

# pen is at the corner of the paper
for i in xrange(np.size(th,0)):
    if i % 2 == 0:
        for j in xrange(np.size(th,1)):
            if th[j, i] == 1:
                sp.Popen(['./penDown.out']).wait()
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./penUp.out']).wait()

            else:
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./straight.out']).wait()

        sp.Popen(['./straight.out']).wait()
        sp.Popen(['./straight.out']).wait()
        sp.Popen(['./straight.out']).wait()
        sp.Popen(['./left90.out']).wait()
        sp.Popen(['./straight.out']).wait()
        sp.Popen(['./left90.out']).wait()
        rasterCtr()
        sp.Popen(['./straight.out']).wait()
        rasterCtr()
        sp.Popen(['./straight.out']).wait()
        rasterCtr()
        sp.Popen(['./straight.out']).wait()

    else:
        for j in xrange(np.size(th,1)):
            if th[j, i] == 1:
                sp.Popen(['./penDown.out']).wait()
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./penUp.out']).wait()

            else:
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./straight.out']).wait()
                sp.Popen(['./straight.out']).wait()

        sp.Popen(['./straight.out']).wait()
        sp.Popen(['./straight.out']).wait()
        sp.Popen(['./straight.out']).wait()
        sp.Popen(['./right90.out']).wait()
        sp.Popen(['./straight.out']).wait()
        sp.Popen(['./right90.out']).wait()
        rasterCtr()
        sp.Popen(['./straight.out']).wait()
        rasterCtr()
        sp.Popen(['./straight.out']).wait()
        rasterCtr()
        sp.Popen(['./straight.out']).wait()

exit()
