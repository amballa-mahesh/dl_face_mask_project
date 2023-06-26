
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils import image_treat
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import matplotlib_inline

best_model = load_model("artifacts/best_model.h5")

df = pd.read_csv('artifacts/test.csv')
image_names = df['file_name']
for i in range(0,3):
  image = load_img('downloads/Images/'+str(image_names[i]),target_size=(224,224)) 
  image_change = image_treat(image)
  print(image_change.shape)
  pred = best_model.predict(image_change).round()
  predict_result=''
  if pred[0] == 1:
    predict_result = 'Image is with Mask'
  else:
    predict_result = 'Image is not with Mask'
  plt.figure(figsize = (2,2))
  plt.imshow(image)
  plt.title(predict_result)
  plt.show()