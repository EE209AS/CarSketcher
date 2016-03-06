import os, sys
import urllib2, urllib
from reconstruct import reproject
from CarCameraControl import CarCameraControl as CCC
import numpy as np

host = 'http://172.20.10.4:8000/'
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
    return 'downloaded image ' + name

def pushControl(ctrl):
    '''
        ctrl: a sequence of control signal
    '''
    url = host + '?Action=Move'
    values = {'name' : 'Sherman',
              'ctrl' : ctrl}
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    res = response.read()
    return res

def capture(name):
    url = host + '?Action=Capture&Name=' + name
    values = {'name' : 'Sherman',
              'ctrl' : []}
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    res = response.read()
    return res

def calcCor():
    css = []
    ccc = CCC()
    for i in range(1,3):
        corners = ccc.Get4Corners('sample' + str(i) + '.jpg')
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

if __name__ == "__main__":
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
        cor = calcCor()
        print pushControl(cor)
    else:
        print 'error command'





