import os, sys
import urllib2, urllib
from reconstruct import reproject
# from CarCameraControl import CarCameraControl as CCC
import numpy as np
# from FindPaperReturnFrame import FindPaperReturnFrame
from FindPaperCornerOnJpg import FindPaperCornerOnJpg as FPC

class ControlError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
class PaperDetectError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
class CaptureError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

#dummy value!!!
camera_height = 21
host = 'http://localhost:8000/'
# translation from camera frame to vehicle frame
R = np.identity(3)
T = np.zeros((3,1))
def getImage(name):
    '''
        name -- name of the image
    '''
    url = host + name
    response = urllib2.urlopen(url)
    f = open(name, 'wb')
    img = response.read()
    f.write(img)
    f.close()
    return 'downloaded ' + name

def pushControl(ctrl):
    '''
        ctrl: a sequence of control signal
    '''
    url = host + '?Action=Move'
    values = {'name' : 'Sherman',
              'ctrl' : ctrl}
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req, timeout=60)
    res = response.read()
    print res
    if res != 'Success':    # control signal failed, very severe!
        raise ControlError(str(ctrl) + 'failed')

def capture(name):
    url = host + '?Action=Capture&Name=' + name
    values = {'name' : 'Sherman',
              'ctrl' : []}
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    res = response.read()
    return  res == 'Success'


def calcCor_prev():
    '''
        for testing phase
    '''
    css = []
    ccc = CCC()

    for i in range(1,3):
        # corners = ccc.Get4Corners('sample' + str(i) + '.jpg')
        frame = cv2.imread('sample' + str(i) + '.jpg')
        fp=FindPaperReturnFrame(ccc.width, ccc.height)
        img, ctr=fp.ReadCameraDataReturnJPG(frame)
        cs = []
        for i in corners:
            x,y = i.ravel()
            cs.append([x,y])
        css.append(cs)
        print cs

    X_cam = reproject('sample2.jpg', 'sample1.jpg', css[1], css[0])[0]  # just take arbibtrary!
    X_car = np.dot(R, X_cam) + T
    print X_car
    return X_car[:2]

def calcCor():
    '''
        for testing phase
    '''
    css = []
    fpc = FPC()
    for i in range(1,3):
        cs = fpc.hasPaper('sample' + str(i) + '.jpg', True)

        if cs is None:
            raise NameError('paper detection failed!')
        css.append(cs)
        print cs
    Xs = reproject('sample2.jpg', 'sample1.jpg', css[1], css[0], camera_height) # just take arbibtrary!
    print Xs
    # what returned is an np array!
    # X_cam = np.reshape(Xs[0], (3,1))
    # X_car = np.dot(R, X_cam) + T
    # return X_car[:2]
    return [-Xs[0][0], Xs[0][2]]
def tooClose(cor):
    if (cor[0]**2 + cor[1]**2 < camera_height**2):
        return np.append(cor, 1)
    return cor

def calculateCoordinate(css):

    Xs = reproject('sample2.jpg', 'sample1.jpg', css[1], css[0], camera_height) # just take arbibtrary!
    X_cam = np.reshape(Xs[0], (3,1))
    X_car = np.dot(R, X_cam) + T
    return X_car[:2]

if __name__ == "__main__":
    '''
        usage: python main.py cap 1 --> cap 2 --> dl 1 --> dl 2 --> calc
    '''

    if sys.argv[1] == 'dl':
        iname = 'sample' + sys.argv[2] + '.jpg'
        print getImage(iname)
    elif sys.argv[1] == 'cap':
        iname = 'sample' + sys.argv[2] + '.jpg'
        print capture(iname)

    elif sys.argv[1] == 'idle':
        print pushControl([-1, -1])
    elif sys.argv[1] == 'calc':
        # ensure sample1 and sample2 are already there!
        if len(sys.argv) == 3 and sys.argv[2] == 'ptest':
            fpc = FPC()
            for i in range(1,3):
                cs = fpc.hasPaper('sample' + str(i) + '.jpg', True)
                if cs is None:
                    raise NameError('paper detection failed!')
            exit()
        # regular test
        cor = calcCor()
        cor = tooClose(cor)
        if len(sys.argv) == 3 and sys.argv[2] == 'test':
            pass
        else :
            pushControl(cor)

    elif sys.argv[1] == 'start':
        # main execution loop
        count = 0;
        inames = ['sample1.jpg', 'sample2.jpg']
        css = [None, None]
        while True:

            try:
                # image capture phase
                iname = inames[count % 2]
                if not capture(iname):
                    raise CaptureError(iname + ' capture failed')
                print getImage(iname)
                cs = fpc.hasPaper('sample' + str(i) + '.jpg')
                if cs is None:
                    raise NameError(iname + 'paper detection failed!')
                # idle phase
                if count % 2 == 0:
                    pushControl([-1,-1])
                # start triangulation
                css[count % 2] = cor
                if not None in css:
                    cor = calculateCoordinate(css)
                    cor = tooClose(cor)
                    pushControl(cor)
                    css = [None, None]

                count += 1

            except CaptureError as e:
                # current image don't work, retake some shit!
                print e.value
                continue
            # paper detection failed -- rotate
            except NameError as e:
                print e.message
                pushControl([-2,-2])
            except ControlError as e:
                print e.value
                break

    else:
        print 'error command'
