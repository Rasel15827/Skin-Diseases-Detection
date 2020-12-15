# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:24:48 2020

@author: Rasel
"""

from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#image_size = 224
test_dir = 'Dataset2/Validation'
test_batchsize = 20


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(112, 150),
        batch_size=test_batchsize,
        shuffle = False,
        class_mode='categorical')

FLOW1_model = load_model('resnet50.h5')

#Confusion Matrix and Classification Report
Y_pred = FLOW1_model.predict_generator(test_generator, test_generator.samples // test_generator.batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['ak', 'bcc', 'bk','df','mel','mn','vasc']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))


#Evaluating using Keras model_evaluate:
x, y = zip(*(test_generator[i] for i in range(len(test_generator))))
x_test, y_test = np.vstack(x), np.vstack(y)
loss, acc = FLOW1_model.evaluate(x_test, y_test, batch_size=32)

print("Accuracy: " ,acc)
print("Loss: ", loss)

# plot the confusion matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plot_confusion_matrix(cm, classes = range(7))
plt.savefig('VGG16_CM')
