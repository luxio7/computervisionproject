# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:40:26 2019

@author: emile
"""
from lxml import etree
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from keras.models import Sequential 
import pickle
from matplotlib import pyplot as plt
import sys
import datetime
from model import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


#%%
# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
#voc_root_folder = "C:/Users/rikp/Desktop/computervisiondatabase/VOCtrainval_11-May-2009/VOCdevkit/"  # please replace with the location on your laptop where you unpacked the tarball
voc_root_folder = "C:/Users/emile/Documents/KUL/ComputerVision/computervisionproject/data/VOCdevkit/"
image_size = 256    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)
                    # neem iets dat deelbaar is door 8
        
#%%            
if False:
    print("step 1")
    # step1 - build list of filtered filenames
    """
    annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []
    for a_f in annotation_files:
        tree = etree.parse(os.path.join(annotation_folder, a_f))
        if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
            filtered_filenames.append(a_f[:-4])
    """
    # step2 - build (x,y) for TRAIN/VAL (classification)
    print("step 2")
    classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation/")
    #classes_files = os.listdir(classes_folder)
    #train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_train.txt' in c_f]
    #val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_val.txt' in c_f]
    
    train_file = os.path.join(classes_folder, 'train.txt')
    val_file = os.path.join(classes_folder, 'val.txt')
        
    def build_classification_dataset(f_cf):
        """ build training or validation set
    
        :param list_of_files: list of filenames to build trainset with
        :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
        """
        temp = []
        train_labels = []
        #for f_cf in list_of_files:
        with open(f_cf) as file:
            temp = file.read().splitlines()
            #temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            #label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            #train_labels.append(len(temp[-1]) * [label_id])
        #train_filter = [item for l in temp for item in l]
    
        image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
        image_filenames = [os.path.join(image_folder, file) for f in temp for file in os.listdir(image_folder) if f in file]
        #image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if f in file]
        x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
            'float32')
        # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
        """
        y_temp = []
        for tf in train_filter:
            y_temp.append([1 if tf in l else 0 for l in temp])
        y = np.array(y_temp)
        """
        segmentation_image_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationClass")
        segmentation_filenames = [os.path.join(segmentation_image_folder, file) for f in temp for file in os.listdir(segmentation_image_folder) if f in file]
        y = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in segmentation_filenames]).astype('float32')
        return x, y
    
    x_train, y_train = build_classification_dataset(train_file)
    print('%i training images with %i segmentated images' %(x_train.shape[0], y_train.shape[0]))
    x_val, y_val = build_classification_dataset(val_file)
    print('%i validation images with %i segmentated images' %(x_val.shape[0], y_val.shape[0]))
    
        # ------------------------
    
    # store output -> dat elke keer maken duurt te lang
    
    # Saving the objects:
    with open('seg_x_train.txt', 'wb') as f:
        pickle.dump(x_train, f)
    with open('seg_y_train.txt', 'wb') as f:
        pickle.dump(y_train, f)
    with open('seg_x_val.txt', 'wb') as f:
        pickle.dump(x_val, f)
    with open('seg_y_val.txt', 'wb') as f:
        pickle.dump(y_val, f)
        
else:
    # Getting back the objects:
    with open('seg_x_train.txt', 'rb') as f:
        x_train = pickle.load(f)
    with open('seg_y_train.txt', 'rb') as f:
        y_train = pickle.load(f)
    with open('seg_x_val.txt', 'rb') as f:
        x_val = pickle.load(f)
    with open('seg_y_val.txt', 'rb') as f:
        y_val = pickle.load(f)
        
    
#%%
    
print("creating model")
model = unet()
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit(x_train, y_train,epochs=300,batch_size = 128, shuffle = True, callbacks=[model_checkpoint])
model.fit(x_train, y_train,epochs=300,batch_size = 128, shuffle = True)

results = model.predict(x_val,32, steps = 1,verbose=1)


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]


COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    
def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

saveResult("results/segmentation/test",results)





sys.exit()