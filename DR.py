
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Arvind Balachandrasekaran
"""
# import keras and numpy libraries
from __future__ import print_function
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from PIL import Image
from keras.callbacks import ModelCheckpoint
#from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import csv

import os.path

# Input parameters
batch_size = 128
num_classes = 5
epochs = 5
num_examples = 20000
# input image dimensions
img_rows, img_cols = 256, 256

i = 0

labels = np.zeros(num_examples)
filenames = [[]]*num_examples

label_counter = np.zeros((num_classes,1))

# Reading the data. Choosing N (4000) examples at most from each class. 
# Trying to train on a subset of the data.
with open('trainLabels.csv', 'rb') as csvfile:
    next(csvfile)
    spamreader = csv.reader(csvfile, dialect='excel')
    for row in spamreader:
        if label_counter[int(row[1])] < num_examples/num_classes:
            filenames[i] = row[0]
            labels[i] = int(row[1])
            label_counter[int(row[1])]+=1
            i+=1
        if i>=num_examples:
            break

num_examples = i
labels = labels[:num_examples]
filenames = filenames[:num_examples]

print('Selected Files')

dataset = np.zeros(shape=(img_rows,img_cols,3,num_examples))

# Resizing and storing the images in a numpy array.
i=0
for filename in filenames:
    path = '../../Machine Learning/DR/train/'+filename+'.jpeg'
    if os.path.exists(path):
        im = Image.open(path)
        im = im.resize((img_rows,img_cols), Image.ANTIALIAS)
        tmp = np.array(im.getdata(),np.float32)
        im_new = tmp.reshape(im.size[1], im.size[0], tmp.shape[1])
        dataset[:,:,:,i] = im_new
        i+=1
        #print('Found '+filename)
    else:
        print('Not Found !!!!!!!!!! '+filename)

# Additional preprocessing. The images from three channels look quite different. 
# The following steps perform color normalisation.        
print('Prepared Data')
dataset[dataset<10]=0
dataset_norm = np.zeros(shape=(img_rows,img_cols,3,num_examples))
for i in range(num_examples):
    I = dataset[:,:,:,i]
    Isum = np.sum(I,axis=2)
    Isum[Isum==0]=1e-6
    Itemp = np.repeat(Isum[:,:,np.newaxis],3,axis=2)
    dataset_norm[:,:,:,i] = np.divide(I,Itemp)
  
# Preparing data for training and validation.    
dataset_norm = dataset_norm.reshape(img_rows*img_cols,3,num_examples)
dataset_norm = np.transpose(dataset_norm,(2,0,1))

indices = np.random.permutation(num_examples)
dataset_norm = dataset_norm[indices,:,:]
labels = labels[indices]

x_train = dataset_norm[:int(0.9*num_examples),:,:]
x_valid = dataset_norm[int(0.9*num_examples):,:,:]

y_train = labels[:int(0.9*num_examples)]
y_valid = labels[int(0.9*num_examples):]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_valid = x_valid.reshape(x_valid.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')

#x_train /= 255
#x_valid /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)


datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,horizontal_flip=True,
    vertical_flip=True)
datagen.fit(x_train)

datagenValid = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)
datagenValid.fit(x_train)

# Setting up the parameters for the architecture.
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01),kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01),kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01),kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01),kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01),kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01),kernel_initializer='glorot_normal'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# save the model corresponding to lowest validation error
modelsave = ModelCheckpoint(
    filepath='retinopathy.h5', monitor='val_acc', save_best_only=True, verbose=1, mode='max')

# Training and saving the intermediate losses
iterations = 400
train_loss = np.zeros((iterations,1))
val_loss = np.zeros((iterations,1))

# the distribution of data is very skewed with class four and five having very few examples. Hence to balance the
# data distribution, I have appropriately weighted the data in each class 
# before training. 
class_weight = {0 : 1.0, 1: 1.63, 2: 1.0, 3: 4.58, 4: 5.64}

for iteration in range(iterations):


    hist=model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                     epochs=epochs,verbose=1,validation_data=datagenValid.flow(x_valid, y_valid), callbacks=[modelsave],class_weight = class_weight )

    #model.save('AB.h5')
    val_loss[iteration-1] = hist.history['val_loss'][0]
    train_loss[iteration-1] = hist.history['loss'][0]
    
    np.save('retinopathyTrainErr',train_loss[:iteration])
    np.save('retinopathyValidErr',val_loss[:iteration])

score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

