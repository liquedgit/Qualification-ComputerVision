import cv2
import os
import numpy as np
from ImageProcess import *
import math

cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_model():
    train_path = './TrainImage'
    facelist = []
    folderList = []
    
    for idx, train_dir in enumerate(os.listdir(train_path)):
        for image_path in os.listdir(f'{train_path}/{train_dir}'):
            path = f'{train_path}/{train_dir}/{image_path}'
            img = cv2.imread(path)
            gray = grayscale_filter(img)
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) < 1:
                continue
            
            for face_rect in faces:
                x, y, w, h = face_rect
                faceimg = gray[y:y+w, x:x+h]
                facelist.append(faceimg)
                folderList.append(idx)
    
    recognizer.train(facelist,np.array(folderList))
    return recognizer

def get_name(idx):
    return os.listdir('./TrainImage')[idx]

faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# recognizer = train_model()

def livecam():
    
    cap = cv2.VideoCapture(0)
    currpreference = 1
    keypress = 0
    while True:
        _, img = cap.read()
        gray = grayscale_filter(img)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if(currpreference == 1):
                currpreference = 2
            elif(currpreference == 2):
                currpreference = 3
            elif currpreference == 3:
                currpreference= 4
            elif currpreference == 4:
                currpreference = 1
            
        for(x, y, w, h) in faces:
            face_part = gray[y: y+w, x:x+h]
            idx,conf = recognizer.predict(face_part)
            conf = math.floor(conf*100)/100
            if conf > 45:
                text = f'{get_name(idx)}'
            else:
                text = 'Unknown'
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
            if currpreference == 1:
                img[y:y+w, x:x+h] = spesific_image_blur(img,x,y,w,h)
            elif currpreference == 2:
                grayimg = grayscale_filter(img[y:y+w, x:x+h])
                colorimg = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)
                img[y:y+w, x:x+h] = colorimg
            elif(currpreference == 3):
                threshimg = threshold_binary(img[y:y+w, x:x+h])
                colorimg = cv2.cvtColor(threshimg, cv2.COLOR_GRAY2BGR)
                img[y:y+w,x:x+h] = colorimg
            elif(currpreference == 4):
                threshimg = threshold_otsu(img[y:y+w, x:x+h])
                colorimg = cv2.cvtColor(threshimg, cv2.COLOR_GRAY2BGR)
                img[y:y+w,x:x+h] = colorimg
            cv2.putText(img, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)


        cv2.imshow('Face Recognition', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
