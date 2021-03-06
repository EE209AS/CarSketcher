import cv2
from CarCameraControl import CarCameraControl
from rasterAdjust import rasterAdjust

cc=CarCameraControl()
cc.InitCamera()
ra=rasterAdjust()
ctr=-2


frame = cc.GetOneVideoFrame()
try:
    ctr, frame = ra.RasterAdjust(frame)
except:
    print "error"

cc.DestoryCamera()

# while(1):
#
#     keypressed=cv2.waitKey(1) & 0xFF
#     if keypressed == ord('q'):
#         break
#
#     frame = cc.GetOneVideoFrame()
#
#     try:
#         ctr, frame = ra.RasterAdjust(frame)
#     except:
#         print "error"
#
#     if ctr == 1:
#         print "turn right"
#     elif ctr == -1:
#         print "turn left"
#     elif ctr == 0:
#         print "go"
#
#     cv2.imshow("test",frame)
#
# cc.DestoryCamera()
# cv2.destroyAllWindows()
#
#
