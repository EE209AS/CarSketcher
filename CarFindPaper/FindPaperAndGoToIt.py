import cv2
from FindPaperReturnFrame import FindPaperReturnFrame
from CarCameraControl import CarCameraControl


cc=CarCameraControl()
cc.InitCamera()

fp=FindPaperReturnFrame(cc.width, cc.height)


while(True):

    keypressed=cv2.waitKey(1) & 0xFF
    if keypressed == ord('q'):
        break
    elif keypressed == ord('p'):
        print toDraw

    frame = cc.GetOneVideoFrame()
    toDraw = cc.CaptureThePicToDraw(frame)
    # buf storage jpg
    buf, ctr=fp.ReadCameraDataReturnJPG(frame)

    cv2.imshow("toDraw", toDraw)


cc.DestoryCamera()
cv2.destroyAllWindows()
