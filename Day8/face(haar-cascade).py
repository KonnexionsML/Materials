import cv2
import numpy as np
import time
#Haar-Cascade Algorithm
#Haar like features
#edges,curves,corners,surfaces.
#fd = cv2.CascadeClassifier(r'full_path')
fd=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
fd1=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye_tree_eyeglasses.xml")
v=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    r,i=v.read()
    print(i)
    img=i[::]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f = fd.detectMultiScale(gray,1.1,7)
    f1= fd1.detectMultiScale(gray,1.1,7)
    #x,y,w,h
    for x,y,w,h in f:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,'Face detected',(x,y),font,1,(255,0,0),2)
    for x,y,w,h in f1:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('face',img)
    k=cv2.waitKey(1)
    if(k==ord('q')):
        break
cv2.destroyAllWindows()
v.release()
        
    
