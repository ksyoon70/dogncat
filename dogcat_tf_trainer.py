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


original_dataset_dir = './datasets/training_set'
original_dataset_cats_dir = os.path.join(original_dataset_dir,'cats')
original_dataset_dogs_dir = os.path.join(original_dataset_dir,'dogs')

categories = ["cat","dog"]

#이미지 크기 조정 크기
IMG_SIZE = 224

training_data = []
training_file_info = []

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
    

def preload_file_name(max_file_num,categories,dirs,file_info):
    for categorie in categories:
        path = os.path.join(dirs,categorie +'s')
        class_num = categories.index(categorie)
        ctfiles = os.listdir(path)
        ctotal_file_num = np.minimum(max_file_num,len(os.listdir(path)))
        files = [ctfiles[i] for i in range(0,ctotal_file_num)]
        for file in files:
            file_info.append([file,class_num])

#------------- 영상 복사 여기까지 --------------
def create_training_data(max_files):

    for categorie in categories:
        path = os.path.join(train_dir,categorie +'s')
        class_num = categories.index(categorie)
        image_files = os.listdir(path)
        imges = [image_files[i] for i in range(0,max_files)]
        for img in imges:
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))  #크기ㅣ 변경함.
                training_data.append([new_array, class_num]) 
            except Exception as e:
                pass

preload_file_name(100,categories,train_dir,training_file_info)

random.shuffle(training_file_info)


"""
create_training_data(1000)

random.shuffle(training_data)

for feature, label in training_data:
    x.append(feature)
    y.append(label)

  
conv_base = VGG16(weights='imagenet',
                  include_top = False,
                  input_shape=(IMG_SIZE,IMG_SIZE,3))
conv_base.summary()


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

model.summary()
"""