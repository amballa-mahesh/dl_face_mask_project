# -*- coding: utf-8 -*-
"""deeplearing_test2_realtime_face_detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ozi4y1NCthDZGTIiJRuisXnPVSOC5Cl_
"""

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
print('libraries loaded')

best_model = load_model(r"C:/Users/Amballa/Desktop/best_model.h5")
  print('model loaded')

face_cascade = cv2.CascadeClassifier(r'C:/Users/Amballa/Desktop/haarcascade_frontalface_default.xml')
print('cascade loaded...')

def image_treat(image):
  image = cv2.resize(image,(224,224))
  image = img_to_array(image)
  image = image/255
  image_change = np.expand_dims(image,axis=0)
  return(image_change)

cap.release()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frames = cap.read()
    frames = cv2.flip(frames,1)
    #cv2.imwrite('camera_image.jpg',frames)
    faces = face_cascade.detectMultiScale(frames,scaleFactor = 1.05, minNeighbors = 5, minSize = (50,50))    
    for(x,y,w,h) in faces:
        image_treated = image_treat(frames)
        pred = best_model.predict(image_treated).round()
        predict_result = ''
        if pred[0] == 1:
            predict_result = 'Mask Identified'
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frames, 'Thanks for wearing!', (x + 12, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            predict_result = 'No Mask Identified'
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frames, 'Wear mask', (x + 12, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        print(predict_result)
        cv2.imshow('image',frames)
        print(frames.shape)
        # print(image_resize.shape)
        
    if cv2.waitKey(3)& 0xff == 27:
            break
cap.release()
cv2.destroyAllWindows()