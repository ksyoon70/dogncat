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
from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model


IMG_SIZE = 224

categories = ["dog","cat"]

test_dir = './datasets/test_set'

model = load_model('cats_and_dogs_small_3.h5')

print('테스트용 이미지 갯수:',len(os.listdir(test_dir)))

if len(os.listdir(test_dir)):

    files = os.listdir(test_dir)
    for file in files:
        
        try:
            img_path = os.path.join(test_dir,file)
            img = image.load_img(img_path,target_size=(IMG_SIZE,IMG_SIZE))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor,axis=0)
            img_tensor /= 255.
            
            preds = model.predict(img_tensor)
            
            if preds[0] > 0.5:
                tilestr = 'prediction:' + 'dog'
            else:
                tilestr = 'prediction:' + 'cat'
        
            plt.title(tilestr)
            plt.imshow(img_tensor[0])
            plt.show()
        except Exception as e:
            pass
                
        
        
        
        





