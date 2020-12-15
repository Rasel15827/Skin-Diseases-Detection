# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.layers import Input, Lambda, Dense, Flatten,Conv2D, Dropout, MaxPooling2D,GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras
from keras import regularizers
# re-size all the images to this
#IMAGE_SIZE = [224, 224]

train_path = 'Dataset2/Train'
valid_path = 'Dataset2/Validation'


  # useful for getting number of classes
folders = glob('Dataset2/Train/*')
# our layers - you can add more if you want

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.1,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=90,
                                   horizontal_flip = True,
                                   vertical_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Dataset2/Train',
                                                 target_size = (112, 150),
                                                 batch_size = 20,
                                                 class_mode = 'categorical'
                                                 )

test_set= test_datagen.flow_from_directory('Dataset2/Validation',
                                            target_size = (112, 150),
                                            batch_size = 20,
                                            class_mode = 'categorical'
                                            )


model_resnet = ResNet50(input_shape=(112,150,3), weights='imagenet', include_top=False)

for layer in model_resnet.layers:
    layer.trainable = True
    
model= Sequential()
model.add(model_resnet)
'''model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu",kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax',kernel_regularizer=regularizers.l2(0.02)))'''
model.add(Conv2D(64, (3, 3), activation = 'relu',padding='same'))
model.add(Dropout(0.40))
model.add(Conv2D(64, (3, 3), activation = 'relu', padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.40))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(7, activation='softmax'))

from keras.optimizers import Adam
#optimizer = Adam(lr=0.0005)


model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# view the structure of the model
model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("resnet50.h5", monitor='val_accuracy', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1, mode='auto')

batch_size= 20

# fit the model
#with tf.device('/device:GPU:0'): 
r = model.fit_generator(
        training_set,
        validation_data= test_set,
        epochs=50,
        steps_per_epoch= training_set.samples//training_set.batch_size,
        validation_steps=test_set.samples//test_set.batch_size,
        callbacks=[checkpoint,early]
     )
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('Resnet_LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('Resnet_AccVal_acc')