# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:33:44 2019

@author: emile
"""
from keras.preprocessing.image import ImageDataGenerator
from model import unet

#BATCH_SIZE = 32
#EPOCHS 

#https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
#https://keras.io/preprocessing/image/
#model = unet()
def train_model(model, x_train, y_train, val_data, model_name = "segmentation.h5", 
                create_new_model = False, batch_s = 2, nb_epochs = 5):
    model_path = "models/"+model_name
    if create_new_model:
        print("creating model")
    else:
        print("loading model")
        model.load_weights(model_path)
        
    #create image augmentation data generator 
    aug = ImageDataGenerator(rotation_range = 0, 
                             zoom_range = 0.05, 
                             width_shift_range = 0.05, 
                             height_shift_range = 0.05, 
                             horizontal_flip = True, 
                             shear_range = 0.02,
                             fill_mode = 'nearest')
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True)

    
    history = model.fit_generator(
            aug.flow(x_train, y_train, batch_size=batch_s),
            steps_per_epoch = len(x_train)//batch_s, 
            epochs = nb_epochs,
            validation_data = val_data,
            callbacks = [model_checkpoint])
    #history = model.fit(x_train, y_train, nb_batches = 16, epoch = 5, verbose = 1, shuffle = True)
    
    #model.save_weights(model_path)
    
    return history
