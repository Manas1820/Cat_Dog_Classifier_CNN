# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 02:53:05 2020

@author: Manas gupta
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()
classifier.add(Conv2D( 32,(3, 3),border_mode='same',input_shape=(64,64,3) ,activation ='relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim =128 , activation='relu'))
classifier.add(Dense(output_dim =1, activation='sigmoid'))
classifier.compile(optimizer='adam' , loss='binary_crossentropy' ,metrics=['accuracy'])

# Loading the dataset

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

train_datagen = ImageDataGenerator(
    rescale =  1./255 ,
    shear_range = 0.2 ,
    zoom_range = 0.2  ,
    horizontal_flip = True
)
test_datagen = ImageDataGenerator (rescale = 1./255)
training_set= train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size =(64,64) ,
    batch_size = 32      ,
    class_mode ='binary'
)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size =(64,64) ,
    batch_size = 32      ,
    class_mode ='binary'
)
classifier.fit_generator(
    training_set ,
    steps_per_epoch = 8000 ,
    epochs = 25 ,
    validation_data = test_set ,
    validation_steps = 2000
)

# Saving the model

classifier.save('classifier.h5')

# loading the model
from keras.models import load_model

classifier = load_model('classifier.h5')

test_image=image.load_img('4001.jpg', target_size = (64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis = 0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction='Dog'
else:
    prediction='Cat'
print(prediction)