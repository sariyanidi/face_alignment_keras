#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:45:22 2021
@author: sariyanidi
"""
from misc import get_landmarks, preprocess_image
import cv2

#
# This code is only for illustrating how the method works.
#
# We DO NOT advise to use this code for inference. Instead, use the code
# in github.com/sariyanidi/face_alignment_opencv, which is maintained
# more frequently and has more flexibility (works for images/videos/image dirs)
# and includes face detection etc. (The lines below assume that the 
# face is cropped appropriately) 
#
#

# The model file, readable by OpenCV. You need to creat this file by
# running python freeze.py
landmark_net = cv2.dnn.readNetFromTensorflow('models/model_FAN_frozen.pb')
image_path = 'samples/test.png'

# read image
im_orig = cv2.imread(image_path)

(im, pt_center, scale) = preprocess_image(im_orig, 67, 67, 181, 213)
p = get_landmarks(im, landmark_net, pt_center, scale)

for ip in range(p.shape[0]):
    cv2.circle(im_orig, (p[ip,0], p[ip,1]), 1, (0, 255, 0), -2)

cv2.imshow("Result", im_orig)
cv2.waitKey(0)
cv2.destroyAllWindows()

