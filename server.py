from BaseHTTPServer import BaseHTTPRequestHandler
import subprocess as sp
import cgi
import urlparse
import json, sys, os, re


maxNumBytes = 100
# f = None
# r = None
# sample = 10
r, w = os.pipe()
f = os.fdopen(r)
class PostHandler(BaseHTTPRequestHandler):
	
	def do_GET(self):       # for cross domain reference

		# parsed_path = urlparse.urlparse(self.path)
		# query = parsed_path[4]
		# if not query:
		# 	return 
		# r = int(query[query.find('id=') + 3:])
		arr = []
		n = 3
		for i in range(0,n):
			line = f.readline()
			if not line:
				break
			arr.append(line.split())		
		# for i, val in enumerate(tmp.split('\n')):
		# 	arr.append(val.split())
		origin = "null"
		for name, value in sorted(self.headers.items()):
			# print name, value
			if name == "origin":
				origin = value
		# print arr
		message = json.dumps(arr)
		self.send_response(200)
		self.send_header('Access-Control-Allow-Origin', origin)         #browser required
		self.send_header('Access-Control-Allow-Methods', 'GET')         #browser required
		self.end_headers()
		self.wfile.write(message)
		return


	def do_POST(self):
		parsed_path = urlparse.urlparse(self.path)
		query = parsed_path[4]
		print 'thisis query: ', query
		form = urlparse.parse_qs(query)
		origin = 'null'
		for name, value in sorted(self.headers.items()):
			# print name, value
			if name == "origin":
				origin = value
		# Begin the response
		self.send_response(200)
		self.send_header('Access-Control-Allow-Origin', origin)         #browser required
		self.send_header('Access-Control-Allow-Methods', 'GET')         #browser required
		self.end_headers()
		
		filename = form["Song"][0] + '.out'
		if form['Action'][0] == 'Start':
			sp.Popen(['./'+filename, str(w)])			
			self.wfile.write('sensor activated!')		#send cookie
			
		elif form['Action'][0] == 'Stop':
			filename2 = form["Song"][0] + '\.out'
			if r is not None:
				os.close(r)
			p1 = sp.Popen(['ps'], stdout=sp.PIPE)
			p2 = sp.Popen(['grep', filename2],stdin=p1.stdout, stdout=sp.PIPE)
			output = p2.communicate()[0]
			pid = output.split()[0]
			print 'killing process', pid
			sp.call(['kill', pid])


		elif form['Action'][0] == 'Tempo':
			sp.call(['./', filename + "_" + "tempo.out"])
		else:
			print 'wrong Action'

		return

from BaseHTTPServer import HTTPServer
server = HTTPServer(('0.0.0.0', 8000), PostHandler)
print 'Starting server on 8000, use <Ctrl-C> to stop'


server.serve_forever()
