# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:40:26 2019

@author: emile
"""
import numpy as np
import os
from skimage import io, color
from skimage.transform import resize
from keras.models import Model
import numpy as np
from keras.models import Sequential 
import pickle
from matplotlib import pyplot as plt
import sys
from model import *
import cv2
from train_with_Generator import *
import datetime
import matplotlib.pyplot as plt


#%%
# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
#voc_root_folder = "C:/Users/rikp/Desktop/computervisiondatabase/VOCtrainval_11-May-2009/VOCdevkit/"  # please replace with the location on your laptop where you unpacked the tarball
voc_root_folder = "C:/Users/emile/Documents/KUL/ComputerVision/computervisionproject/data/VOCdevkit/"
image_size = 256    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)
                    # neem iets dat deelbaar is door 8
create_new_model = False
#%%            
def prepareSegm5ClassesGrayData():
    # step2 - build (x,y) for TRAIN/VAL (classification)
    print("step 2")
    classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
    classes_files = os.listdir(classes_folder)
    train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_train.txt' in c_f]
    val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_val.txt' in c_f]
    
    seg_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation")
    seg_train_file = os.path.join(seg_folder, 'train.txt')
    seg_val_file = os.path.join(seg_folder, 'val.txt')
    def build_segmentation_data(list_of_files, seg_file):
        """ build training or validation set
    
        :param list_of_files: list of filenames to build trainset with
        :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
        """
        temp = []
        train_labels = []
        for f_cf in list_of_files:
            with open(f_cf) as file:
                lines = file.read().splitlines()
                temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
                label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
                train_labels.append(len(temp[-1]) * [label_id])
        train_filter = [item for l in temp for item in l]
        
        #segmentation
        with open(seg_file) as s_f:
            seg_lines = s_f.read().splitlines()      
        seg_train_filter = [item for item in train_filter if item in seg_lines]
    
        image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
        image_filenames = [os.path.join(image_folder, file) for f in seg_train_filter for file in os.listdir(image_folder) if f in file]
        x = np.array([process_x_values(img_f) for img_f in image_filenames]).astype(
            'float32')
        segmentation_image_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationClass")
        segmentation_filenames = [os.path.join(segmentation_image_folder, file) for f in seg_train_filter for file in os.listdir(segmentation_image_folder) if f in file]
        y = np.array([process_segmentation_y_values(img_f) for img_f in segmentation_filenames]).astype('float32')
        return x, y
        
    def process_x_values(img_file):
        gray = color.rgb2gray(io.imread(img_file))
        return resize(gray, (image_size, image_size, 1))

    
    def process_segmentation_y_values(img_file):
        gray = color.rgb2gray(io.imread(img_file))
        ret, binary_img = cv2.threshold(gray, 0.001, 1, cv2.THRESH_BINARY)
        return resize(binary_img, (image_size, image_size, 1))
    
    x_train, y_train = build_segmentation_data(train_files, seg_train_file)
    print('%i training images with %i segmentated images' %(x_train.shape[0], y_train.shape[0]))
    x_val, y_val = build_segmentation_data(val_files, seg_val_file)
    print('%i validation images with %i segmentated images' %(x_val.shape[0], y_val.shape[0]))
    
        # ------------------------
    
    # store output -> dat elke keer maken duurt te lang
    
    # Saving the objects:
    with open('data/ProcessedData/seg_x_train_5_classes_gray.txt', 'wb') as f:
        pickle.dump(x_train, f)
    with open('data/ProcessedData/seg_y_train.txt_5_classes_gray.txt', 'wb') as f:
        pickle.dump(y_train, f)
    with open('data/ProcessedData/seg_x_val.txt_5_classes_gray.txt', 'wb') as f:
        pickle.dump(x_val, f)
    with open('data/ProcessedData/seg_y_val.txt_5_classes_gray.txt', 'wb') as f:
        pickle.dump(y_val, f)
    return x_train, y_train, x_val, y_val

        
def loadSegm5ClassesGrayData():
    # Getting back the objects:
    with open('data/ProcessedData/seg_x_train_5_classes_gray.txt', 'rb') as f:
        x_train = pickle.load(f)
    with open('data/ProcessedData/seg_y_train_5_classes_gray.txt', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/ProcessedData/seg_x_val_5_classes_gray.txt', 'rb') as f:
        x_val = pickle.load(f)
    with open('data/ProcessedData/seg_y_val_5_classes_gray.txt', 'rb') as f:
        y_val = pickle.load(f)
    return x_train, y_train, x_val, y_val
    

