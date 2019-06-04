# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:48:41 2019

@author: emile
"""
from model import unet
import cv2
import os
from skimage import io, color

MODEL_PATH = "models/segmentations.h5"

model = unet(pretrained_weights=MODEL_PATH)

def saveResult(save_path,npyfile,thresholded = True, threshold = 0.5):
    if thresholded:
        save_path = save_path + "/binary"
    for i,item in enumerate(npyfile):
        res = item[:,:,0].astype('uint8')
        if thresholded:    
            ret, res = cv2.threshold(res, threshold, 1, cv2.THRESH_BINARY)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),res)

#CODE HERE
results = model.predict(x_val,1,verbose=1)
saveResult("results",results, False)
