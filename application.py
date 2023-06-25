from flask import Flask, render_template,url_for,Response
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
print('libraries loaded')

best_model = load_model(r"artifacts/best_model.h5")
print('model loaded')

face_cascade = cv2.CascadeClassifier(r'artifacts/haarcascade_frontalface_default.xml')
print('cascade loaded...')

def image_treat(image):
  image = cv2.resize(image,(224,224))
  image = img_to_array(image)
  image = image/255
  image_change = np.expand_dims(image,axis=0)
  return(image_change)


app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frame():
    while True:
        success,frames = camera.read()
        frames = cv2.flip(frames,1)    
        faces = face_cascade.detectMultiScale(frames,scaleFactor = 1.05, minNeighbors = 5, minSize = (50,50))    
        for(x,y,w,h) in faces:
            image_treated = image_treat(frames)
            pred = best_model.predict(image_treated).round()
            predict_result = ''
            if pred[0] == 0:
                predict_result = 'Mask Identified'
                cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frames, 'Thanks for wearing!', (x + 12, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                predict_result = 'No Mask Identified'
                cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frames, 'Wear mask', (x + 12, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            print(predict_result)

        if not success:
            break
        else:
            ret,buffer = cv2.imencode('.jpg',frames)
            frames = buffer.tobytes()

        yield(b'--frames\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frames +b'\r\n')

   



@app.route('/')
def index():    
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frames')
    
    



if __name__ =="__main__":
    app.run(debug=True)
