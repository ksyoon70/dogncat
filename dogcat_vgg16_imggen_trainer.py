#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:22:21 2021

@author: headway
"""
import numpy as np
import os, shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random
from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


"""
#GPU 사용시 풀어 놓을 것
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""

original_dataset_dir = './datasets/training_set'
original_dataset_cats_dir = os.path.join(original_dataset_dir,'cats')
original_dataset_dogs_dir = os.path.join(original_dataset_dir,'dogs')


categories = ["cat","dog"]

#이미지 크기 조정 크기
IMG_SIZE = 224
#배치 싸이즈
BATCH_SIZE = 20

training_data = []
training_file_info = []

validation_data = []
validation_file_info = []


x = []
y = []


#------------- 영상 생성 및 디렉토리 시작 --------------

base_dir = './datasets/cats_and_dogs_small'
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'validation')
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir,'cats')
if not os.path.isdir(train_cats_dir):
    os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'dogs')
if not os.path.isdir(train_dogs_dir):
    os.mkdir(train_dogs_dir)
    

validation_cats_dir = os.path.join(validation_dir,'cats')
if not os.path.isdir(validation_cats_dir):
    os.mkdir(validation_cats_dir)


validation_dogs_dir = os.path.join(validation_dir,'dogs')
if not os.path.isdir(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)
    
test_cats_dir = os.path.join(test_dir,'cats')
if not os.path.isdir(test_cats_dir):
    os.mkdir(test_cats_dir)


test_dogs_dir = os.path.join(test_dir,'dogs')
if not os.path.isdir(test_dogs_dir):
    os.mkdir(test_dogs_dir)
    
if not len(os.listdir(train_cats_dir)):
    
    fnames = ['cat.{}.jpg'.format(i) for i in range(1,1001)]
    
    for fname in fnames:
        src = os.path.join(original_dataset_cats_dir,fname)
        dst = os.path.join(train_cats_dir,fname)
        shutil.copy(src,dst)
        
else:
    print('훈련용 고양이 이미지 갯수:',len(os.listdir(train_cats_dir)))
 
if not len(os.listdir(validation_cats_dir)):
    
    fnames = ['cat.{}.jpg'.format(i) for i in range(1001,1501)]
    
    for fname in fnames:
        src = os.path.join(original_dataset_cats_dir,fname)
        dst = os.path.join(validation_cats_dir,fname)
        shutil.copy(src,dst)
else:
    print('검증용 고양이 이미지 갯수:',len(os.listdir(validation_cats_dir)))
    
if not len(os.listdir(test_cats_dir)):

    fnames = ['cat.{}.jpg'.format(i) for i in range(1501,2001)]
    for fname in fnames:
        src = os.path.join(original_dataset_cats_dir,fname)
        dst = os.path.join(test_cats_dir,fname)
        shutil.copy(src,dst)
else:
    print('테스트용 고양이 이미지 갯수:',len(os.listdir(validation_cats_dir)))
    
    
if not len(os.listdir(train_dogs_dir)):

    fnames = ['dog.{}.jpg'.format(i) for i in range(1,1001)]
    
    for fname in fnames:
        src = os.path.join(original_dataset_dogs_dir,fname)
        dst = os.path.join(train_dogs_dir,fname)
        shutil.copy(src,dst)
    
else:
    print('훈련용 강아지 이미지 갯수:',len(os.listdir(train_dogs_dir)))
    
if not len(os.listdir(validation_dogs_dir)):
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1001,1501)]
    
    for fname in fnames:
        src = os.path.join(original_dataset_dogs_dir,fname)
        dst = os.path.join(validation_dogs_dir,fname)
        shutil.copy(src,dst)

else:
    print('검증용 강아지 이미지 갯수:',len(os.listdir(validation_dogs_dir)))
    
if not len(os.listdir(test_dogs_dir)):

    fnames = ['dog.{}.jpg'.format(i) for i in range(1501,2001)]
    for fname in fnames:
        src = os.path.join(original_dataset_dogs_dir,fname)
        dst = os.path.join(test_dogs_dir,fname)
        shutil.copy(src,dst)
else:
    print('테스트용 강아지 이미지 갯수:',len(os.listdir(test_dogs_dir)))
    
train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)

#train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMG_SIZE,IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')


validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    target_size=(IMG_SIZE,IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')


  
conv_base = VGG16(weights='imagenet',
                  include_top = False,
                  input_shape=(IMG_SIZE,IMG_SIZE,3))
#conv_base.summary()


# Convolution Layer를 학습되지 않도록 고정 
for layer in conv_base.layers:
    layer.trainable = False 


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

#model.summary()

history = model.fit_generator(train_generator,
                              steps_per_epoch=50,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=25)
 
    
model.save('cats_and_dogs_small_3.h5')



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label ='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label ='Training loss')
plt.plot(epochs, val_loss, 'b', label ='Validation loss')

plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.show()

