import os, sys
import urllib2, urllib

host = 'http://localhost:8000/'

def getImage(name):
    '''
        name -- name of the image
    '''    
    url = host + name
    response = urllib2.urlopen(url)
    return response.read()

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



