import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import os
import sys
from src.logger import logging
from src.exception import CustomException
import tkinter
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import applications
from utils import image_treat
from tensorflow.keras.models import load_model
logging.info('libraries loaded....')

os.chdir('e:/Vscode_files/End_to_End_Ml_project_Facemask')
logging.info('dir changed to default...')

df = pd.read_csv('artifacts/train.csv')
logging.info('read the train.csv')
print(df.head())
df['labels'] = df['labels'].astype('str')


train_df,test_df = train_test_split(df,test_size=0.2,stratify=df['labels'],random_state=43)
logging.info('data splitted as train and test')

train_data = ImageDataGenerator(rescale = 1./255,
                                rotation_range =20,
                                horizontal_flip= True,
                                vertical_flip = True)


test_data =ImageDataGenerator(rescale =1./255)

train_data_generator = train_data.flow_from_dataframe(dataframe = train_df,
                                                      directory='downloads/Images/',
                                                      target_size =(224,224),
                                                      x_col = 'file_name',
                                                      y_col = 'labels',
                                                      color_mode = 'rgb',
                                                      class_mode = 'binary',
                                                      batch_size = 32,
                                                      seed = 42,
                                                      shuffle = True,
                                                      validate_filenames = True
                                                      )

test_data_generator = test_data.flow_from_dataframe(dataframe = test_df,
                                                      directory='downloads/Images/',
                                                      target_size =(224,224),
                                                      x_col = 'file_name',
                                                      y_col = 'labels',
                                                      color_mode = 'rgb',
                                                      class_mode = 'binary',
                                                      batch_size = 32,
                                                      seed = 42,
                                                      shuffle = True,
                                                      validate_filenames = True
                                                      )

logging.info('train and test data generators are created..')

base_model = tensorflow.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                                       weights='imagenet',
                                                       include_top=False
                                                       )
base_model.trainable = False
base_model.summary()
logging.info('MobileNetV2 model downloaded is completed..')

flatten_layer = tensorflow.keras.layers.Flatten()
output_layer  = tensorflow.keras.layers.Dense(1, activation = 'sigmoid')

final_model = tensorflow.keras.Sequential([base_model,flatten_layer,output_layer])

logging.info('final model created')

mc = tensorflow.keras.callbacks.ModelCheckpoint(filepath='artifacts/best_model.h5',
                                                save_best_only=True,
                                                mode='auto',
                                                monitor='val_loss'
                                                )

es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=5,
                                              mode = 'auto',
                                              restore_best_weights=True)

logging.info('model check point and early stopping is defined...')

final_model.compile(optimizer = 'adam',
                    metrics= 'accuracy',
                    loss = 'binary_crossentropy'
                    )
logging.info('final model complied')

history = final_model.fit(train_data_generator,
                          steps_per_epoch=train_data_generator.samples//32,
                          validation_data=test_data_generator,
                          validation_steps=test_data_generator.samples//32,
                          epochs = 20,
                          callbacks=[mc,es])

logging.info('final model training completed and save as best model in artifacts folder...')

history.history

print('Training Accuracy Score: ',np.mean(history.history['accuracy']).round(2))
print('Validation Accuracy Score: ',np.mean(history.history['val_accuracy']).round(2))
print('Training Loss: ',np.mean(history.history['loss']).round(2))
print('Validation Loss: ',np.mean(history.history['val_loss']).round(2))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend('Training Accuracy Score')
plt.legend('Validation Accuracy Score')
plt.title('Accuracy State')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend('Training Loss Score')
plt.legend('Validation Loss Score')
plt.title('Loss State')
plt.show()

logging.info('Training completed sucessfully.............')
