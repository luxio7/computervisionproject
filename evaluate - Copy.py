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
import matplotlib.pyplot as plt
from keras.models import load_model


MODEL_PATH = "models/segmentationBroken.hdf5"

model = load_model(MODEL_PATH)

def saveResult(save_path,npyfile,thresholded = True, threshold = 0.5):
    if thresholded:
        save_path = save_path + "/binary"
    for i,item in enumerate(npyfile):
        res = item[:,:,0]
        if thresholded:    
            ret, res = cv2.threshold(res, threshold, 1, cv2.THRESH_BINARY)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),res)


def showResult(index, threshold = 0.2):
        fig=plt.figure(figsize = (10,10))
        fig.add_subplot(221)
        plt.imshow(x_val[index][:,:])
        fig.add_subplot(222)
        plt.imshow(y_val[index][:,:,0])
        fig.add_subplot(223)
        res = results[index][:,:,0]
        plt.imshow(res)
        fig.add_subplot(224)
        ret, t_res = cv2.threshold(res, threshold, 1, cv2.THRESH_BINARY)
        plt.imshow(t_res)
        
def diceScore(ground, result):
    dice = np.sum([result[ground==1]])*2.0 / (np.sum(result) + np.sum(ground))
    return dice

def computeBestThreshold(results, y_val):
    diceScores = []
    for thresh in [0.15,0.18,0.2,0.22,0.25,0.28,0.31,0.4,0.5]:
        binary_results = thresholdResults(results,thresh)
        diceScores.append(diceScoreAverage(y_val, binary_results))
    return diceScores

"""
:param y_val: array of size (nb_images, size, size, 1)
:param bin_results: list of binary segmentation image of size(nb_images, size, size)
"""
def diceScoreAverage(y_val, bin_results):
    return np.mean([diceScore(y_val[i][:,:,0],bin_results[i]) for i in range(len(bin_results))])

def thresholdResults(results, threshold = 0.5):
    res = []
    for img in results:
        ret, binary = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
        res.append(binary)
    return res
        
        
#CODE HERE
results = model.predict(x_val,1,verbose=1)
binary_results = thresholdResults(results, 0.3)
saveResult("results",results, False)
print("diceScoreAverage: "+str(diceScoreAverage(y_val,binary_results)))

