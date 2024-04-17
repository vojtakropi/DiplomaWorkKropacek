import random
import numpy as np
import tensorflow as tf
import keras
from keras.activations import gelu
from keras.models import Model
from keras.layers import Concatenate, MaxPool2D, Conv2D, Input, UpSampling2D, Dense

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


# Image size
SIZE = 512
INPUT_SHAPE = (SIZE, SIZE, 1)


def CNN(INPUT_SHAPE = INPUT_SHAPE):

    inputs = Input(INPUT_SHAPE, name = 'input')

    conv1 = Conv2D(filters=16, kernel_size=(3,3), stride=1)(inputs)
    maxpool1 = MaxPool2D(pool_size=2, strides=2)(conv1)
    conv17 = Conv2D(filters=32, kernel_size=(3, 3), stride=1)(maxpool1)
    maxpool17 = MaxPool2D(pool_size=2, strides=2)(conv17)
    conv18 = Conv2D(filters=64, kernel_size=(3, 3), stride=1)(maxpool17)
    maxpool18 = MaxPool2D(pool_size=2, strides=2)(conv18)
    conv19 = Conv2D(filters=64, kernel_size=(3, 3), stride=1)(maxpool18)
    maxpool19 = MaxPool2D(pool_size=2, strides=2)(conv19)
    conv20 = Conv2D(filters=96, kernel_size=(3, 3), stride=1)(maxpool19)
    maxpool20 = MaxPool2D(pool_size=2, strides=2)(conv20)
    conv2 = Conv2D(filters=256, kernel_size=(3, 3), stride=1)(maxpool20)
    maxpool2 = MaxPool2D(pool_size=2, strides=2)(conv2)
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), stride=1)(maxpool2)
    gel1 = gelu(conv3)
    conv4 = Conv2D(filters=384, kernel_size=(3, 3), stride=1)(gel1)
    gel2 = gelu(conv4)
    conv5 = Conv2D(filters=384, kernel_size=(3, 3), stride=1)(gel2)
    gel3 = gelu(conv5)
    conv6 = Conv2D(filters=384, kernel_size=(3, 3), stride=1)(gel3)
    gel4 = gelu(conv6)
    fully_c = Dense(units=500)(gel4)
    fully_c2 = Dense(units=1)(fully_c)

    model = Model(inputs=inputs, outputs=fully_c2, name='CNN')

    return model
