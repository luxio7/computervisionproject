# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:37:10 2019

@author: emile
"""

def showResult(index, thresholded=True, threshold = 0.5):
        fig=plt.figure(figsize = (10,10))
        fig.add_subplot(221)
        plt.imshow(x_val[index])
        fig.add_subplot(222)
        plt.imshow(y_val[index][:,:,0])
        fig.add_subplot(223)
        res = results[index][:,:,0]
        if thresholded:
            ret, res = cv2.threshold(res, threshold, 1, cv2.THRESH_BINARY)
        plt.imshow(res)


def save_history(history, model_name="segmentation"):
        tijd = datetime.datetime.now()
        tijdstring = tijd.strftime("%H:%M:%S").replace(":", "_")
        path = "models/history_"+model_name+tijdstring
        with open(path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)