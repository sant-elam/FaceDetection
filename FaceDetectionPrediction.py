# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:15:38 2021

@author: Santosh
"""

import mediapipe as mp
import cv2 as cv

from cv2 import cv2

import os 
import glob
import numpy as np

from sklearn.model_selection import train_test_split


from keras.utils import np_utils

import matplotlib.pyplot as plt

import tensorflow as tf


def get_bounding_rectangle(image,region, color, radius):
    
    height, width = image.shape[0:2]
    
    bounding_box = region.location_data.relative_bounding_box   
                 
    left = bounding_box.xmin
    top = bounding_box.ymin
    
    right = left + bounding_box.width
    bottom = top + bounding_box.height
    
    left = (int)(left * width)
    top = (int)(top * height)
    
    if left <0:
        left =0
    if top <0:
        top = 0
    
    right = (int)(right * width)
    bottom = (int)(bottom * height)
    

    
    return left, top, right, bottom


def drw_bounding_rectangle(left, top, right, bottom, image, color, str_Mask):
    
    cv.rectangle(image, (left, top), (right, bottom), color, 1)
    
    cv.line(image, (left, top), (left, top+10), color, 3)
    cv.line(image, (left, top), (left+10, top), color, 3)
    
    cv.line(image, (right, top), (right, top+10), color, 3)
    cv.line(image, (right, top), (right-10, top), color, 3)
    
    
    cv.line(image, (left, bottom), (left, bottom-10), color, 3)
    cv.line(image, (left, bottom), (left+10, bottom), color, 3)
    
    cv.line(image, (right, bottom), (right, bottom-10), color, 3)
    cv.line(image, (right, bottom), (right-10, bottom), color, 3)
    
    cv.putText(image, str_Mask, (left, top-5), cv.FONT_HERSHEY_SIMPLEX, 0.5,  color,  2)
    
    
def convert_img_np(image_resize, target_size):
    
    image_s = np.array(image_resize, dtype=object)    
    image_s = image_s.reshape(1, target_size, target_size, 3)
    image_s = image_s.astype('float32')     
    image_s /= 255
    
    return image_s



COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)

face_detection = mp.solutions.face_detection


capture = cv.VideoCapture(0)


MODEL_SELECTION = 1
CONFIDENCE = 0.5

TARGET_SIZE = 256

detection = face_detection.FaceDetection(model_selection=MODEL_SELECTION, 
                                         min_detection_confidence= CONFIDENCE)

model_path = "C:/MEDIA_PIPE/FACE_MASK/FaceMask.h5"
model = tf.keras.models.load_model(model_path)


while True:
    results, image = capture.read()
    
    if results:
        
        image_convert = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        outputs = detection.process(image_convert)

        if outputs.detections:

             for region in outputs.detections:   
                                
                left, top, right, bottom = get_bounding_rectangle(image,region, COLOR_GREEN, 2)
                
                crop_image = image[top:bottom, left:right]
                

                height, width = crop_image.shape[0:2]
                           
                image_resize = cv.resize(crop_image, (TARGET_SIZE, TARGET_SIZE))

                image_s = convert_img_np(image_resize, TARGET_SIZE)
                
                detect_class = model.predict_classes(image_s)

                if detect_class == 0:
                    color = COLOR_GREEN
                    str_Mask = 'With Mask'
                else:
                    color = COLOR_RED
                    str_Mask = 'Without Mask'
                
                drw_bounding_rectangle(left, top, right, bottom, image, color, str_Mask)

                
                
        cv.imshow("Face_Detection", image)     
       
        if cv.waitKey(30) & 255 == 27:
           break
        
capture.release()
cv.destroyAllWindows()

