# -*- coding: utf-8 -*-
"""
@author: Rasel
"""

from keras.layers import Input, Lambda, Dense, Flatten,GlobalAveragePooling2D, Dropout, MaxPooling2D, Conv2D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras
from keras import regularizers
import numpy as np
from glob import glob

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

# re-size all the images to this
IMAGE_SIZE = [299, 299]

train_path = 'Dataset2/Train'
valid_path = 'Dataset2/Validation'


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

# add preprocessing layer to the front of VGG
inception =InceptionV3(input_shape=(112,150,3), weights='imagenet', include_top=False)

    
# don't train existing weights
for layer in inception.layers:
  layer.trainable = True
  
  # useful for getting number of classes
folders = glob('Dataset2/Train/*')

'''x = Flatten()(inception.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=inception.input, outputs=prediction)'''

model = Sequential()
model.add(inception)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu",kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax',kernel_regularizer=regularizers.l2(0.02)))
model.summary()


from keras.optimizers import Adam
opt = Adam (lr=0.0001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


#creating checkpoint
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("inceptionV3.h5", monitor='val_accuracy', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

batch_size=20

# fit the model
with tf.device('/device:GPU:0'): 
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
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')