## Imports + seed

# Common
import os
import sys
import random

import cv2
import numpy as np
from glob import glob
import tensorflow as tf

# Images 
import cv2 as cv

# Augmentation
import albumentations as A
from tqdm import tqdm
SIZE = 512

def preprocess_image(image):
    img = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    # img = cv.bitwise_not(img)    # INVERT THE IMAGE
    img = np.expand_dims(img, axis=-1)
    if (type(img) == tf.uint16):
        img /= 65535.0
    elif (type(img) == tf.uint8):
        img /= 255.0

    return np.array(img)


def image_augmentation():
    seed = 2019
    random.seed = seed
    np.random.seed = seed
    tf.seed = seed
    # Image size
    SIZE = 512
    INPUT_SHAPE = (SIZE, SIZE, 1)
    print(tf.__version__)

    path = ''
    source_image_path = path + 'train/source/'
    target_image_path = path + 'train/target2/'

    source_val_image_path = path + 'val/source/'
    target_val_image_path = path + 'val/target/'

    # Get Images
    source_image_names = sorted(glob(source_image_path + "*.png"))
    target_image_names = sorted(glob(target_image_path + "*.png"))
    print(source_image_names)
    print(target_image_names)

    source_val_image_names = sorted(glob(source_val_image_path + "*.png"))
    target_val_image_names = sorted(glob(target_val_image_path + "*.png"))

    print(len(source_image_names))
    print(len(target_image_names))

    print(len(source_val_image_names))
    print(len(target_val_image_names))

    print(source_image_names[0], target_image_names[0])
    print(source_val_image_names[0], target_val_image_names[0])

    train_transform = A.Compose(
        [
            A.GaussNoise(p=0.3),
            A.MultiplicativeNoise(p=0.15),
            A.CoarseDropout(2, 80, 80, 1, 40, 40, p=0.3)
        ],
    )

    test_transform = A.Compose(
        [
            A.InvertImg(p=1.0),
            # A.CLAHE(clip_limit=1.6, tile_grid_size=(10,10), p=1.0),
        ],
        additional_targets={'image0': 'image'}
    )

    transform = A.Compose(
        [
            A.InvertImg(),
            A.CLAHE(clip_limit=[1, 2], tile_grid_size=(10, 10), p=0.6),
            A.UnsharpMask(p=0.35),
            A.RandomCropFromBorders(p=0.6),
            A.ElasticTransform(p=0.25),
            A.GridDistortion(p=0.25),
            A.Perspective(p=0.35),
            A.HorizontalFlip(p=0.35),
            A.ShiftScaleRotate(p=0.7, shift_limit_y=0.1, shift_limit_x=0.1),
            A.RandomBrightnessContrast(p=0.35),
        ],
        additional_targets={'image0': 'image'}
    )

    val_transform = A.Compose(
        [
            A.InvertImg(),
            A.CLAHE(clip_limit=[1, 2], tile_grid_size=(10, 10), p=0.6),
            A.UnsharpMask(p=0.35),
            A.HorizontalFlip(p=0.45),
            A.RandomBrightnessContrast(p=0.45),
        ],
        additional_targets={'image0': 'image'}
    )

    target_train_path = path + "augmented3/train/"
    target_val_path = path + "augmented3/val/"

    for times in range(10):
        for i in tqdm(range(len(source_image_names))):
            image = cv.imread(source_image_names[i], 0)
            image0 = cv.imread(target_image_names[i], 0)
            image = cv.resize(image, (SIZE, SIZE), cv.INTER_CUBIC)
            image0 = cv.resize(image0, (SIZE, SIZE), cv.INTER_CUBIC)
            image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
            image0 = cv.normalize(image0, None, 0, 255, cv.NORM_MINMAX)

            train_trans = train_transform(image=image)
            trans = transform(image=train_trans['image'], image0=image0)

            image = cv.resize(trans['image'], (SIZE, SIZE), cv.INTER_CUBIC)
            image0 = cv.resize(trans['image0'], (SIZE, SIZE), cv.INTER_CUBIC)

            image = np.array(image)
            image0 = np.array(image0)
            cv.imwrite(target_train_path + "/source/" + source_image_names[i].split('/')[-1][:-4] + "_" + str(
                times) + ".png", image)
            cv.imwrite(target_train_path + "/target/" + source_image_names[i].split('/')[-1][:-4] + "_" + str(
                times) + ".png", image0)

    for times in range(10):
        for i in tqdm(range(len(source_val_image_names))):
            image = cv.imread(source_val_image_names[i], 0)
            image0 = cv.imread(target_val_image_names[i], 0)

            image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
            image0 = cv.normalize(image0, None, 0, 255, cv.NORM_MINMAX)

            trans = val_transform(image=image, image0=image0)

            image = cv.resize(trans['image'], (SIZE, SIZE), cv.INTER_CUBIC)
            image0 = cv.resize(trans['image0'], (SIZE, SIZE), cv.INTER_CUBIC)

            image = np.array(image)
            image0 = np.array(image0)

            cv.imwrite(target_val_path + "/source/" + source_val_image_names[i].split('/')[-1][:-4] + "_" + str(times) + ".png",
                       image)
            cv.imwrite(target_val_path + "/target/" + source_val_image_names[i].split('/')[-1][:-4] + "_" + str(times) + ".png",
                       image0)


if __name__ == "__main__":
    image_augmentation()
