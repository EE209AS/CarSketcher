# CarSketcher

## Overall procedure
1, Power up the car and face the car to the paper  
2, The Edison board and the user computer connected to same LAN  
3, Open webpage and input the IP address of the Edison board  
4, User press the StartCaptureImg button and the Camera on the car should stream back the video to the webpage  
5, User control the car to capture the picture he want to draw  
6, User press StartFindPaper button and the car firstly approach the paper by using FindPaperReturnFrame  
7, During the car approaching, it does serveral 3D reconstruction  
8, When the car get close (<30cm) to the paper, it start approach the paper by using the info gathered through 3d recon  
9, Then the car should move to one corner of the paper and send back to webpage that the car is ready to draw  
10, The user press StartDraw button and the car start drawing  
