# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
import numpy as np
from glob import glob

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

# re-size all the images to this
IMAGE_SIZE = [112, 150] 

train_path = 'Dataset2/Train'
valid_path = 'Dataset2/Validation'


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
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
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = True

  # useful for getting number of classes
folders = glob('Dataset2/Train/*')

'''# our layers - you can add more if you want
x = Flatten()(vgg.output)


# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
'''
model = keras.Sequential([vgg,
                          keras.layers.Flatten(),
                          keras.layers.Dense(units=256, activation='relu'),
                          keras.layers.Dense(units=256, activation='relu'),
                          keras.layers.Dense(len(folders), activation='softmax')])


#specifying optimizer
# opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# tell the model what cost and optimization method to use

from keras.optimizers import Adam
opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


# view the structure of the model
model.summary()

#creating checkpoint

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_0.h5", monitor='val_accuracy', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

batch_size =20
# fit the model
with tf.device('/device:GPU:0'): 
    r = model.fit_generator(
        training_set,
        validation_data= test_set,
        epochs = 50,
        steps_per_epoch= training_set.samples//training_set.batch_size,
        validation_steps=test_set.samples//test_set.batch_size,
        callbacks=[checkpoint,early]
     )
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('VGG_LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('VGG_AccVal_acc')

'''score = model.evaluate(test_set, verbose=0)
ev= score[1]*100
print('Test accuracy:'+' %.2f' % ev+ "%")
import tensorflow as tf

from keras.models import load_model
model.save('skinDiseases.h5')'''
