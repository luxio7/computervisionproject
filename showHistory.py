# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:32:13 2019

@author: emile
"""
import pickle
import matplotlib.pyplot as plt

#1 epoch
#path = 'history_segmentation12_28_18'
#15 epochs
#path = 'history_segmentation14_27_37'
#path = 'history_segmentation16_11_35'
#path = 'history_segmentation19_46_02'
#path = 'history_segmentation20_28_16'
#path = 'history_segmentation21_09_48'

def showHist(history):
    print(history.keys())
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

loss = []
val_loss = []
acc = []
val_acc = []
for path in ['history_segmentation12_28_18','history_segmentation14_27_37','history_segmentation16_11_35',
             'history_segmentation19_46_02','history_segmentation20_28_16','history_segmentation21_09_48',
             'history_segmentation22_59_59','history_segmentation23_27_47']:
    with open('models/'+path, 'rb') as f:
            history = pickle.load(f)
    loss= loss + history['loss']
    val_loss = val_loss + history['val_loss']
    acc = acc + history['acc']
    val_acc = val_acc + history['val_acc']         

plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()