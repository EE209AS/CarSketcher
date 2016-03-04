import cv2
import numpy as np

cap = cv2.VideoCapture(0)
count=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,4,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(frame,(x,y),5,(0,0,255),-1)

    edges = cv2.Canny(frame,100,200)


    # Display the resulting frame
    cv2.imshow('frame',edges)
    keypressed=cv2.waitKey(1) & 0xFF
    if keypressed == ord('q'):
        break
    elif keypressed == ord('t'):
        count=count+1
        filename = "/Users/YingnanWang/Desktop/test%d.png"%count
        print filename
        cv2.imwrite(filename,frame)

# When everything done, release the ccapture
cap.release()
cv2.destroyAllWindows()
