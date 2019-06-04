# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:31:17 2019

@author: emile
"""
from model import unet
import segmentation_5_classes_gray

model = unet(input_size=(256,256,1))
model.summary()
x_train,y_train, x_val, y_val = segmentation_5_classes_gray.prepareSegm5ClassesGrayData()
history = train_model(model, x_train, y_train, (x_val, y_val), nb_epochs = 1, create_new_model=True)

save_history(history)
