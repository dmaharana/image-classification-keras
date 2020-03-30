# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# https://blog.francium.tech/build-your-own-image-classifier-with-tensorflow-and-keras-dc147a15e38e

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
from keras.layers import *
from keras.optimizers import *

# %matplotlib auto

trainImageDirs = [
    '../image-data/train']
testImageDirs = [
    '../image-data/test1']

modelFile = './model-data/model.hdf5'


def list_images(imgdirs):
    imageData = []
    for imgdir in imgdirs:
        for fn in os.listdir(imgdir):
            imageData.append(os.path.join(imgdir, fn))

    # shuffle(imageData)
    return imageData


def one_hot_label(img):
    label = os.path.split(img)[-1].split('.')[0]
    #label = img.split('.')[0]
    label_dict = {
        'cat': np.array([1, 0]),
        'dog': np.array([0, 1])
    }
    ohl = np.array([0, 0])

    for key in label_dict:
        if key in label:
            ohl = label_dict[key]
            break

    # print('filename: {}, ohl: {}'.format(img, ohl))
    return ohl


def read_image(img):
    # print('reading image: {}'.format(img))
    dimg = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    dimg = cv2.resize(dimg, (64, 64))

    return ([np.array(dimg), one_hot_label(img)])


def model_creation(modelFile, tr_img_data, tr_lbl_data):
    print('Creating model')
    model = Sequential()
    epoches = 30
    # keras will internally add batch dimention
    model.add(InputLayer(input_shape=[64, 64, 1]))
    model.add(Conv2D(filters=32, kernel_size=5, strides=1,
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=50, kernel_size=5, strides=1,
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=80, kernel_size=5, strides=1,
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, activation='softmax'))
    optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists(modelFile):
        model.load_weights(modelFile)
    else:
        model.fit(x=tr_img_data, y=tr_lbl_data, epochs=epoches, batch_size=100)
        model.summary()
        model.save(modelFile)

    return model


def plot_results(trainingImageList, testingImageList):
    testingImages = []
    shuffle(testingImageList)

    print('Reading test data')
    for imgName in testingImageList[1000:1020]:
        testingImages.append(read_image(imgName))

    #testingData = np.array([i[0] for i in testingImages]).reshape(-1,64,64,1)
    #testingLabelData = np.array([i[1] for i in testingImages])

    if os.path.exists(modelFile):
        print('reading model from file: {}'.format(modelFile))
        trainingData = []
        trainingLabelData = []
    else:
        print('Creating new model')
        trainingImages = []
        shuffle(testingImageList)
        for imgName in trainingImageList:
            trainingImages.append(read_image(imgName))
        trainingData = np.array(
            [i[0] for i in trainingImages]).reshape(-1, 64, 64, 1)
        trainingLabelData = np.array([i[1] for i in trainingImages])

    model = model_creation(modelFile, trainingData, trainingLabelData)

    fig = plt.figure(figsize=(14, 14))

    for cnt, data in enumerate(testingImages):
        y = fig.add_subplot(6, 5, cnt+1)
        img = data[0]
        data = img.reshape(1, 64, 64, 1)
        model_out = model.predict([data])

        print('model predict: ', model_out)
        if np.argmax(model_out) == 1:
            str_label = 'dog'
        else:
            str_label = 'cat'

        y.imshow(img, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    fig.savefig('output.png')


def prepare_data(imgdir, preString):
    #imgdir = '/home/titu/Pictures/img-data/test-new/cars'
    #preString = 'car'

    for fileName in os.listdir(imgdir):
        newFileName = '{}.{}'.format(preString, fileName)
        print('rename {} to {}'.format(fileName, newFileName))
        os.rename(os.path.join(imgdir, fileName),
                  os.path.join(imgdir, newFileName))


def main():
    trainingImageList = list_images(trainImageDirs)
    testingImageList = list_images(testImageDirs)

    #print("trainingImageList: {}", trainingImageList)
    #print("testingImageList: {}", testingImageList)

    plot_results(trainingImageList, testingImageList)


main()
