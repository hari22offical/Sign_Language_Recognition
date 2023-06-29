
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time



offset=20
imagesize=300

folder='C:/Users/harin/Pictures/Sign Language Recognition/Sign Language Recognition/Dataset/Z'
counter=0

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)

while True:
    
    
    success, img = cap.read()
    hands, img =detector.findHands(img)
    
    if hands:
        
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgWhite=np.ones((imagesize,imagesize, 3),np.uint8)*255
        imgcrop= img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgcropShape=imgcrop.shape
        
        aspectRatio=h/w
        
        if aspectRatio > 1:
            
            k = imagesize/h
            wCal=math.ceil(k*w)
            imageResize=cv2.resize(imgcrop, (wCal, imagesize))
            imageResizeShape=imageResize.shape
            wgap=math.ceil((imagesize-wCal)/2)
            imgWhite[:,wgap:wCal+wgap]=imageResize
            
            
        else:
              
              k = imagesize/w
              hCal=math.ceil(k*h)
              imageResize=cv2.resize(imgcrop, ( imagesize, hCal))
              imageResizeShape=imageResize.shape
              hgap=math.ceil((imagesize-hCal)/2)
              imgWhite[hgap:hCal+hgap , : ]=imageResize 
              
        cv2.imshow("Imagecrop",imgcrop)
        cv2.imshow("imgWhite",imgWhite) 
        
        
    cv2.imshow("Image",img)    
    key=cv2.waitKey(1)
    
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', imgWhite)
        print(counter)