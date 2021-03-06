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


original_dataset_dir = './datasets/training_set'
original_dataset_cats_dir = os.path.join(original_dataset_dir,'cats')
original_dataset_dogs_dir = os.path.join(original_dataset_dir,'dogs')

categories = ["cat","dog"]

#이미지 크기 조정 크기
IMG_SIZE = 224
#배치 싸이즈
BATCH_SIZE = 4

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
    

def preload_file_name(max_file_num,categories,dirs,file_info):
    for categorie in categories:
        path = os.path.join(dirs,categorie +'s')
        class_num = categories.index(categorie)
        ctfiles = os.listdir(path)
        ctotal_file_num = np.minimum(max_file_num,len(os.listdir(path)))
        files = [ctfiles[i] for i in range(0,ctotal_file_num)]
        for file in files:
            file_info.append([os.path.join(path,file),class_num])

#------------- 영상 복사 여기까지 --------------


def _create_training_data(filename,label):
 
    img_array = cv2.imread(str(filename))
    resize_array = cv2.resize(img_array,(224,224))  #크기ㅣ 변경함.
    #training_data.append([resize_array, label])  # 0 ~ 255 사이의 데이터
    np_image_data = np.asarray(resize_array)
    img_array = tf.convert_to_tensor(np_image_data, dtype=tf.float32) / 255.
    
    return (np_image_data, label)

def _create_training_data1(filename,label):
    img_array = tf.io.read_file(filename)
    img_array = tf.image.decode_jpeg(img_array,channels=3)
    img_array = tf.image.convert_image_dtype(img_array, tf.float32) 
    resize_array = tf.image.resize(img_array,(IMG_SIZE,IMG_SIZE))
    #resize_array /= 255.
    #resize_array = tf.reshape(resize_array, [-1, IMG_SIZE, IMG_SIZE, 3])
    return (resize_array,label)

    
    

preload_file_name(1000,categories,train_dir,training_file_info)

random.shuffle(training_file_info)


preload_file_name(500,categories,validation_dir,validation_file_info)

random.shuffle(validation_file_info)

 #파일이름과 레이블 분리
filenames =[]
labels =[]

for info in training_file_info:
    filenames.append(info[0])
    labels.append(info[1])

vfilenames =[]
vlabels =[]

for info in validation_file_info:
    vfilenames.append(info[0])
    vlabels.append(info[1])


# step 3: parse every image in the dataset using `map`
"""
def _parse_function(filename, label):
    image_reader = tf.WholeFileReader()
    _, image_string = image_reader.read(filename)
    #image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    resized_image = tf.image.resize_images(image, [IMG_SIZE, IMG_SIZE])
    return resized_image, label
"""

#x = create_training_data(filenames[0],labels[0])





training_data = tf.data.Dataset.from_tensor_slices((filenames, labels))

#_create_training_data1(filenames[0],labels[0]) #test

training_data = training_data.map(_create_training_data1)


training_data = training_data.batch(BATCH_SIZE)



# 데이터 증식
"""
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

"""

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

#print(str(type(training_data)))  #데이터 type 확인 

"""
x = training_data.map(lambda x, y: x)
y = training_data.map(lambda x, y: y)
    
datagen.fit(x)
"""


validation_data = tf.data.Dataset.from_tensor_slices((vfilenames, vlabels))

#_create_training_data1(vfilenames[0],vlabels[0]) #test

validation_data = validation_data.map(_create_training_data1)

validation_data = validation_data.batch(BATCH_SIZE)


# step 4: create iterator and final input tensor
"""
for data, label in dataset.take(1):
    proto_tensor = tf.make_tensor_proto(data[0,0])
    array = tf.make_ndarray(proto_tensor)
    plt.imshow(array)
"""


#create_training_data(1000)

#random.shuffle(training_data)
"""
for feature, label in training_data:
    x.append(feature)
    y.append(label)
"""
  
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

history = model.fit(training_data,
            batch_size = BATCH_SIZE,
            validation_data=validation_data,
            validation_steps=50,
            epochs=10,
)
 
    
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