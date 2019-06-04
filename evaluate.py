# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:48:41 2019

@author: emile
"""
from model import unet
import cv2
import os
from skimage import io, color
import numpy as np

MODEL_PATH = "models/segmentation.hdf5"

model = load_model(MODEL_PATH)

def saveResult(save_path,npyfile,thresholded = True, threshold = 0.5):
    if thresholded:
        save_path = save_path + "/binary"
    for i,item in enumerate(npyfile):
        res = item[:,:,0]
        if thresholded:    
            ret, res = cv2.threshold(res, threshold, 1, cv2.THRESH_BINARY)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),res)


def showResult(index, thresholded=True, threshold = 0.5):
        fig=plt.figure(figsize = (10,10))
        fig.add_subplot(221)
        plt.imshow(x_val[index][:,:,0])
        fig.add_subplot(222)
        plt.imshow(y_val[index][:,:,0])
        fig.add_subplot(223)
        res = results[index][:,:,0]
        if thresholded:
            ret, res = cv2.threshold(res, threshold, 1, cv2.THRESH_BINARY)
        plt.imshow(res)
#CODE HERE
results = model.predict(x_val,1,verbose=1)
saveResult("results",results, False)


def diceScore(ground, result):
    dice = np.sum([ground==1])*2.0 / (np.sum(result) + np.sum(ground))
    return dice
