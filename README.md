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


## TODO !!!!!  
#### Webpage

* Rearrange the web page. The web page should contain:
  - IP address input
  - Start button for overall 
  - Textview to show the car current status and give instr to user
  - Video stream back
  - Car control buttons to control the car to the desired position to capture picture
  - Button to capture the image for sketcher
  - Image on the web page to show what will draw on the paper
  - Button to control the car start find the paper and approach it
  - Button to start drawing on the paper
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
