import tensorflow
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from src.logger import logging
from src.exception import CustomException


camera = cv2.VideoCapture(0)
best_model = load_model(r"artifacts/best_model.h5")
face_cascade = cv2.CascadeClassifier(r'artifacts/haarcascade_frontalface_default.xml')


def image_treat(image):
  image = img_to_array(image)
  image = image/255
  image_change = np.expand_dims(image,axis=0)
  return(image_change)

def image_processing(image):
  image = cv2.resize(image,(224,224))
  image = img_to_array(image)
  image = image/255
  image_change = np.expand_dims(image,axis=0)
  return(image_change)

def generate_frame():
    while True:
        success,frames = camera.read()
        frames = cv2.flip(frames,1)    
        faces = face_cascade.detectMultiScale(frames,scaleFactor = 1.05, minNeighbors = 5, minSize = (50,50))    
        for(x,y,w,h) in faces:
            image_treated = image_processing(frames)
            pred = best_model.predict(image_treated).round()
            predict_result = ''
            if pred[0] == 1:
                predict_result = 'Mask Identified'
                cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frames, 'Thanks for wearing..!', (x+2, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                predict_result = 'No Mask Identified'
                cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frames, 'Please Wear mask..', (x + 12, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            print(predict_result)
            logging.info("the is result- "+predict_result)


        if not success:
            logging.info('error occured with capturing device..try(1,2...)')
            break
        else:
            ret,buffer = cv2.imencode('.jpg',frames)
            frames = buffer.tobytes()

        yield(b'--frames\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frames +b'\r\n')
