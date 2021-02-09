#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 08:21:14 2021

@author: headway
"""

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


original_dataset_dir = './datasets/training_set'
original_dataset_cats_dir = os.path.join(original_dataset_dir,'cats')
original_dataset_dogs_dir = os.path.join(original_dataset_dir,'dogs')

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
    
    
    
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#model.summary()

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')


validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

model.save('cats_and_dogs_small_1.h5')
