import cv2                                                
import numpy as np                                    
import subprocess as sp                                    
                                                    
class CarCameraControl:                               
                                                           
    width=[]                                               
    height=[]                                         
    cap=[]                                                 
                                         
    def __init__(self):                               
        pass                                               
                                                    
                                                      
    # Camera Control                                       
    def InitCamera(self):                           
        self.cap = cv2.VideoCapture(0)                
        self.width=self.cap.get(3)                         
        self.height=self.cap.get(4)                 
                                                      
    def DestroyCamera(self):                              
        self.cap.release() 

    def GetOneVideoFrame(self):                           
        ret, frame = self.cap.read()                  
        return frame                                       
                                                    
    def ResizeAndFindEdge(self, oriframe):            
        edge = cv2.Canny(oriframe,100,200)                 
        # resize to 64 * 48                                
        edge = cv2.resize(edge, (64,48))
        return edge                                        
car = CarCameraControl()                              
                                                      
#InitCamera()                                              
#frame = GetOneVideoFrame()                           
# code to create a rectangle image                    
#img = np.zeros((512,512,3), np.uint8)                      
#cv2.line(img,(0,0),(511,511),(255,0,0),5)           
#cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)      
#edge = car.ResizeAndFindEdge(img)                          
#ret,th=cv2.threshold(edge,127,255,cv2.THRESH_BINARY)
                                                      
for i in range(1,64):                                     
        for j in range(1,48):                       
                if (j==1||j==64||i==1||i==48):        
   			sp.Popen(['./penDown.out']).wait()
                        print "pen down"                   
                        sp.Popen(['./straight.out']).wait()
			print "move straight"             
                        sp.Popen(['./penUp.out']).wait()
			print "pen up"                
                else:       
	           	sp.Popen(['./penUp.out']).wait()   
                        print "pen up"                               
                        print "move straight"              
                        sp.Popen(['./straight.out']).wait()
        if(i%2==0):                                        
                print "move left90"   
		sp.Popen(['./left90.out']).wait()                     
                print "move straight"             
		sp.Popen(['./straight.out']).wait()        
                print "move left90"               
		sp.Popen(['./left90.out']).wait()         
        else:                                             
                print "move right90"
		sp.Popen(['./right90.out']).wait()                       
                print "move straight"
		sp.Popen(['./straight.out']).wait()                      
                print "move right90"  
		sp.Popen(['./right90.out']).wait()
