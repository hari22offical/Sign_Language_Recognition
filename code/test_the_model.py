
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time



offset=20
imagesize=300

folder='C:/Users/harin/Pictures/Sign Language Recognition/Sign Language Recognition/Dataset/Z'
counter=0

labels=["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M",
        "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier=Classifier("C:/Users/harin/Pictures/Sign Language Recognition/Sign Language Recognition/models/keras_model.h5",
                      "C:/Users/harin/Pictures/Sign Language Recognition/Sign Language Recognition/models/labels.txt")

while True:
    
    
    success, img = cap.read()
    imgOutput= img.copy()
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
            prediction , index = classifier.getPrediction(imgWhite)        
            print(prediction,index)
            
            
        else:
              
              k = imagesize/w
              hCal=math.ceil(k*h)
              imageResize=cv2.resize(imgcrop, ( imagesize, hCal))
              imageResizeShape=imageResize.shape
              hgap=math.ceil((imagesize-hCal)/2)
              imgWhite[hgap:hCal+hgap , : ]=imageResize
              prediction , index = classifier.getPrediction(imgWhite)      
              
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)      
        cv2.imshow("Imagecrop",imgcrop)
        cv2.imshow("imgWhite",imgWhite) 
        
        
    cv2.imshow("Image",imgOutput)    
    cv2.waitKey(1)
    
   
