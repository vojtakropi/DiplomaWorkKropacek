## Classes and methods for data extraction

import cv2 as cv
import keras
import numpy as np
from random import random
import os
from keras.utils import img_to_array

seed = 2019
np.random.seed = seed

IMAGE_SIZE = 512
BATCH_SIZE = 5

def preprocess_image(image):
    img = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    if random() < .5:
        img = cv.bitwise_not(img)    # INVERT THE IMAGE WITH 50 % PROBABILITY
    return img

## DATA GENERATOR

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, RGB=False):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.RGB = RGB
        self.on_epoch_end()
    
    def __load__(self, id_name):
        image_path = self.path+ "orig/" + id_name
        val_image_path = self.path+ "label/" + id_name.replace("orig", "bonesupp")
        
        if self.RGB:
            img = cv.imread(image_path)
        else:
            img = cv.imread(image_path,0)
            img = np.expand_dims(img, axis = -1)

        
        if self.RGB:
            val_img = cv.imread(val_image_path)
        else:
            val_img = cv.imread(val_image_path,0)
            val_img = np.expand_dims(val_img, axis = -1)



        
        if (val_img.shape != (IMAGE_SIZE, IMAGE_SIZE, 1)):
            val_img = cv.resize(val_img, (IMAGE_SIZE, IMAGE_SIZE))

        if (img.shape != (IMAGE_SIZE, IMAGE_SIZE, 1)):
            img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        
        val_img = img_to_array(val_img)
        img = img_to_array(img)
        
        # Normalization of the images    
        img = img / 255.0
        val_img = val_img / 255.0

        return img, val_img
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        files_batch = self.ids[index*self.batch_size: (index+1)*self.batch_size]
        
        image = []
        val_image = []

        
        for id_name in files_batch:
            _img, _val_img = self.__load__(id_name)
            image.append(_img)
            val_image.append(_val_img)
        image = np.array(image)
        val_image = np.array(val_image)
        
        return image, val_image
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

def getIds(path):    
    ids = os.listdir(path)
    print(len(ids))
    ids = sorted(ids)
    return ids
