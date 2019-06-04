# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:31:17 2019

@author: emile
"""
from model import unet
import segmentation_5_classes_gray
from util import save_history
from train_with_Generator import train_model

x_train,y_train, x_val, y_val = segmentation_5_classes_gray.loadSegm5ClassesGrayData()
history = train_model(x_train, y_train, (x_val, y_val), nb_epochs = 10, create_new_model=False)

save_history(history)
