#sudo modprobe bcm2835-v4l2 -> Activating Raspberry Pi camera for openCV

import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import requests
import os

GPIO.setmode(GPIO.BCM)
    
GPIO_TRIGGER=18
GPIO_ECHO=24

GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def face_eye_detect_module():
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == False:
        print("Video Capturing object initialization failed")
    else:
        print("Video Capturing object initialization Successfull")

    while 1:
        #capturing frame by frame
        ret, img = cap.read()
        
        if ret == False:
            print("Frame is Empty")
        else:
            print("Frame is non-empty")
    
        #operation on frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if(len(faces) == 1):
            face_detect = True
            #print("Face Detected Successfully")
        else:
            #print("Face not Detected")
            face_detect = False

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if(len(eyes) == 1):
                #print("Eyes Detected Successfully")
                eyes_detect = True
            else:
                #print("Eyes not Detected")
                eyes_detect = False
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        cap.release()
        cv2.destroyAllWindows()
        
        return eyes_detect
    

def distance_measure_module():
    
    GPIO.output(GPIO_TRIGGER, True)
    
    time.sleep(0.00001)
    
    GPIO.output(GPIO_TRIGGER, False)
    
    StartTime = time.time()
    StopTime = time.time()
    
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
        
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
        
    TimeElapsed = StopTime - StartTime
    
    distance = (TimeElapsed * 34300) / 2
    
    return distance

def send_sms_module():
    url = "https://www.fast2sms.com/dev/bulk"
    payload = "sender_id=FSTSMS&message=Switching off TV in 15 sec&language=english&route=p&numbers=YOUR_MOBILE_NUMBER"
    headers = {
    'authorization': "*****Authorization Key(You will get it once you register yourself on fast2sms.com)*****",
    'Content-Type': "application/x-www-form-urlencoded",
    'Cache-Control': "no-cache",
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    print(response.text)
    
while 1:
    detected = face_eye_detect_module()
    dist = distance_measure_module()
    if (detected == True) and (dist < 50):
        print("Measured Distance = %.1f cm" % dist)
        send_sms_module()
        time.sleep(15)
        os.system("xset dpms force off")
        time.sleep(60)
        os.system("xset dpms force on") 
    else:
        print("Watching TV from Safe Distance")
