# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 14:01:53 2019

@author: emile
"""

from skimage import io, color

img = io.imread("data/VOCdevkit/VOC2009/JPEGImages/2007_000042.jpg")
segm = io.imread("data/VOCdevkit/VOC2009/SegmentationClass/2007_000042.png")

gray = color.rgb2gray(segm)
ret, binary_img = cv2.threshold(gray, 0.001, 1, cv2.THRESH_BINARY)
res = resize(binary_img, (image_size, image_size, 1))

io.imshow(img)
io.imshow(segm)
io.imshow(gray)
io.imshow(binary_img)
io.imshow(res)

bla = resize(binary_img, (image_size, image_size))
blab = res[:,:,0]
io.imshow(blab)
