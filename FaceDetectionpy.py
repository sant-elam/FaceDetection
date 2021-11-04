# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 00:01:43 2021

@author: 
"""
'''
 FACE DETECTION
 
'''

import mediapipe as mp
import cv2 as cv


face_detection = mp.solutions.face_detection
draw_utils = mp.solutions.drawing_utils
draw_color = draw_utils.DrawingSpec((0,255,0), thickness=3, circle_radius = 1)


video_path = "C:/Users/Santosh/OneDrive/Desktop/FACE_DETECTION/VID-20211028-WA0001.mp4"

capture = cv.VideoCapture(0)

MODEL_SELECTION = 0
CONFIDENCE = 0.7

detection = face_detection.FaceDetection(model_selection=MODEL_SELECTION, 
                                         min_detection_confidence= CONFIDENCE)


while True:
    results, image = capture.read()
    
    if results:
        image_convert = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        outputs = detection.process(image_convert)
        
        #regions = outputs.detections
        
        if outputs.detections:
            for region in outputs.detections:
                draw_utils.draw_detection (image,
                                           region,
                                           draw_color,
                                           draw_color)
                
        cv.imshow("Face_Detection", image)     
        
        if cv.waitKey(30) & 255 == 27:
            break
        
capture.release()
cv.destroyAllWindows()