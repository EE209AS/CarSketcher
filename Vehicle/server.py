from BaseHTTPServer import BaseHTTPRequestHandler
import subprocess as sp
import cgi,time
import urlparse
import json, sys, os, re
from CarCameraControl import CarCameraControl as CCC
from FindPaperReturnFrame import FindPaperReturnFrame

maxNumBytes = 100

class PostHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):       # for cross domain reference

        parsed_path = urlparse.urlparse(self.path)
        path = parsed_path[2]
        # if not query:
        #   return 
        # r = int(query[query.find('id=') + 3:])
        origin = "null"
        for name, value in sorted(self.headers.items()):
            # print name, value
            if name == "origin":
                origin = value
        
        # send image
        fname = os.path.basename(path)
        # print fname
        f = open(fname, 'rb')
        img = f.read()
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', origin)         #browser required
        self.send_header('Access-Control-Allow-Methods', 'GET')         #browser required
        self.send_header('Content-Type', 'image/' + self.findFormat(fname))
        self.end_headers()
        self.wfile.write(img)
        return

    def findFormat(self, fname):
        index = fname.index('.')
        if fname[index + 1:] == 'jpg':
            return 'jpeg'
        elif fname[index + 1:] == 'png':
            return 'png'
        else:
            raise NameError('unknown image format')

    def do_POST(self):

        _form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST'

                     })
        parsed_path = urlparse.urlparse(self.path)
        query = parsed_path[4]
        # print 'this is query: ', query
        form = urlparse.parse_qs(query)
        ## set cross domain header
        origin = 'null'
        for name, value in sorted(self.headers.items()):
            # print name, value
            if name == "origin":
                origin = value
        self.send_response(200)        
        self.send_header('Access-Control-Allow-Origin', origin)         #browser required
        self.send_header('Access-Control-Allow-Methods', 'POST')         #browser required
        self.end_headers()
        ## Edison control 
        # capture the image and save as the specific name
        if 'Action' in form and form['Action'][0] == 'Capture':
            # # dummy
            # f = open('sample.jpg', 'rb')
            # img = f.read()
            # f.close()
            camera = CCC()
            camera.InitCamera()
            fname = form['Name'][0]
            if 'sample' in fname:
                frame = camera.GetOneVideoFrame()
                img = camera.convert2jpg(frame)
            elif 'bolder' in fname:
                frame = camera.GetOneVideoFrame()
                fp=FindPaperReturnFrame(camera.width, camera.height)
                img, ctr=fp.ReadCameraDataReturnJPG(frame)
            else:
                return   
            #print frame
            
            # write file
            f2 = open(form['Name'][0], 'wb')
            f2.write(img)
            f2.close()
            camera.DestroyCamera()
            self.wfile.write('Success')       
            # self.send_header('Content-Type', 'image/jpeg')  
        elif 'Action' in form and form['Action'][0] == 'Move':
            ctrl = _form["ctrl"].value 
            print 'control accepted: ', ctrl
            force = 1
            if len(ctrl) < 3:
                force = 0
            sp.call(['RobotControl/papercontrol.out', str(ctrl[0]), str(ctrl[1]), str(force)])
            # time.sleep(30)
            self.wfile.write('Success')

        else:
            print 'wrong Action'
                
        return

from BaseHTTPServer import HTTPServer
server = HTTPServer(('0.0.0.0', 8000), PostHandler)
print 'Starting server on 8000, use <Ctrl-C> to stop'

server.serve_forever()
