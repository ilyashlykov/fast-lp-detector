#!/usr/bin/env python
# coding: utf-8

# In[85]:


import os
import cv2
import sys
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.color import rgb2gray
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import itertools
import glob
import tensorflow as tf
import requests,io
from datetime import datetime
if len(sys.argv) == 1:
    img_path='/Users/ilyashlykov/Downloads/train'
else:
    img_path=sys.argv[1]
def read_img(path):
    filelist = os.listdir(path)
    all_img=[]
    for item in filelist:
        if item.endswith(('.jpg','.bmp','png','jpeg')):
            src = os.path.join(os.path.abspath(path), item)
            src_img = cv2.imread(src)
            all_img.append(src_img)
    return all_img
tich_file=read_img(img_path)
letters=['0' ,'1' ,'2','3' ,'4' ,'5', '6', '7', '8','9', 'A', 'B', 'C', 'E' ,'H','K', 'M','O', 'P', 'T', 'X', 'Y']
def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret
elapsed_time=[]
for i in range(len(tich_file)):
    print('auto #'+str(i+1))
    start_time=datetime.now()
    image0 = tich_file[i]
    image_height, image_width, _ = image0.shape
    image = cv2.resize(image0, (1024,1024))
    image = image.astype(np.float32)
    paths='./model_resnet.tflite'
    interpreter = tf.lite.Interpreter(model_path=paths)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X_data1=np.float32(image.reshape(1,1024, 1024,3))
    input_index = (interpreter.get_input_details()[0]['index'])
    interpreter.set_tensor(input_details[0]['index'], X_data1)
    interpreter.invoke()
    detection = interpreter.get_tensor(output_details[0]['index'])
    net_out_value2 = interpreter.get_tensor(output_details[1]['index'])
    net_out_value3 = interpreter.get_tensor(output_details[2]['index'])
    net_out_value4 = interpreter.get_tensor(output_details[3]['index'])
    img = image0
    razmer=img.shape

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converts from one colour space to the other
    img3=img[:,:,:]

    box_x =int( detection[0,0,0] * image_height)
    box_y = int(detection[0,0,1] * image_width)
    box_width =int( detection[0,0,2] * image_height)
    box_height = int(detection[0,0,3] * image_width)
    if  np.min(detection[0,0,:])>=0:
        cv2.rectangle(img2, ( box_y,box_x), (box_height,box_width ), (230, 230, 21), thickness=5)

        #plt.imshow(img2)
        #plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
        #plt.show()
        image = img3[box_x:box_width,box_y:box_height,:]
        grayscale = rgb2gray(image)
        edges = canny(grayscale, sigma=3.0)
        out, angles, distances = hough_line(edges)
        _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
        angle=np.mean(np.rad2deg(angles_peaks))
        if 0 <= angle <= 90:
            rot_angle = angle - 90
        elif -45 <= angle < 0:
            rot_angle = angle - 90
        elif -90 <= angle < -45:
            rot_angle = 90 + angle
        if abs(rot_angle)>20:
            rot_angle=0
        rotated = rotate(image, rot_angle, resize=True)*255
        rotated =rotated.astype(np.uint8)
        rotated1=rotated[:,:,:]
        minus=np.abs(int(np.sin(np.radians(rot_angle))*rotated.shape[0]))
        if rotated.shape[1]/rotated.shape[0]<2 and minus >10:

            rotated1=rotated[minus:-minus,:,:]
        lab= cv2.cvtColor(rotated1, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        paths='./model1_nomer.tflite'
        interpreter = tf.lite.Interpreter(model_path=paths)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
#         img =rotated1
        img =final # лучше работает при плохом освещении

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128,64))
        img = img.astype(np.float32)
        img /= 255
        
        img1=img.T
        img1.shape
        X_data1=np.float32(img1.reshape(1,128, 64,1))
        input_index = (interpreter.get_input_details()[0]['index'])
        interpreter.set_tensor(input_details[0]['index'], X_data1)

        interpreter.invoke()

        net_out_value = interpreter.get_tensor(output_details[0]['index'])
        pred_texts = decode_batch(net_out_value)
        print('number:',pred_texts)
    else:
        plt.imshow(image0)
        plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
        plt.show() 
        print('нет')
    elapsed_time.append((datetime.now()-start_time).total_seconds())
    print('time:',elapsed_time[i],'s')
print('\nTotal time:','%.5f' %sum(elapsed_time),'s')
print('Average time:','%.5f' %np.mean(elapsed_time),'s')

