#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:06:49 2023

@author: parth
"""

import numpy as np
import os 
import pandas as pd 
import natsort
from natsort import natsorted
from keras.preprocessing.image import ImageDataGenerator
from skimage import io 
import cv2
from PIL import Image
import matplotlib.pyplot as plt

cwd = os.getcwd()

cwd = os.chdir(cwd + '/Desktop/carla_stuff/')

cwd = os.getcwd()

dataset_folder = "dataset"
dataset_dir = os.path.join(cwd, dataset_folder)

labels = []
raw_images = []

for root, dirs, files in os.walk(dataset_dir, topdown=False):
    for filename in files:
        filepath = root + "/" + filename
        
        if 'raw_images' in filepath:
            # im = io.imread(filepath)
            raw_images.append(filepath)
            
        elif 'labels' in filepath:
            # im = io.imread(filepath)
            labels.append(filepath)

raw_images = natsorted(raw_images)
labels = natsorted(labels)

raw_images_set, labels_set = [], []

for img in raw_images:
    img = io.imread(img)
    raw_images_set.append(img)

for img in labels:
    img = io.imread(img)
    labels_set.append(img)    

raw_images_set = np.array(raw_images_set)
raw_images_set = (raw_images_set.astype('float32')) / 255

labels_set = np.array(labels_set)        
labels_set = (labels_set.astype('float32')) / 255

datagen = ImageDataGenerator(horizontal_flip=True)

# -------------------------------------------------------------------- #

## For one image 

# x = io.imread('1.jpg')

# x = x.reshape((1,) + x.shape)
# i=0

# for batch in datagen.flow(raw_images_set, labels_set, batch_size=4,
#                           save_to_dir= 'augmented/raw_images',
#                           save_format='png'):
#     i += 1
#     if i > 19:
#         break 

x = next(datagen.flow(raw_images_set, batch_size=32, seed=1337))
y = next(datagen.flow(labels_set, batch_size=32, seed=1337))


# -------------------------------------------------------------------- #

