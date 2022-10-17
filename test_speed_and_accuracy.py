#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import sys
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
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
import requests, io
from datetime import datetime
import json
import PIL
import time
import base64
from alive_progress import alive_bar
import dvc.api
from tqdm import tqdm

params=dvc.api.params_show()

LEVENSTEIN_TRESHOLD_ERROR = params['Levenstein_const']['levenstein_error']
LEVENSTEIN_TRESHOLD_MISSED = params['Levenstein_const']['levenstein_missed']
NUMBER_WIDTH = params['image_size']['width']
NUMBER_HEIGHT = params['image_size']['height']
HOUGH_TRANSFORM = params['hough']
model_resnet_path = params['paths']['model_resnet']
model_nomer_paths = params['paths']['model_nomer']


parser = argparse.ArgumentParser()
parser.add_argument('-M', '--metrics', type=str, help='path to the file with metrics')
parser.add_argument('-P', '--path', type=str, help='path to dataset folder')
args = parser.parse_args()
img_path = args.path


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


# Вычисление расстояния Ливенштейна по алгоритму Вагнера-Фишера
def levenstein(str_1, str_2):
    n, m = len(str_1), len(str_2)
    D = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            if (i * j == 0):
                D[i][j] = i + j
            else:
                D[i][j] = min(D[i][j - 1] + 1, D[i - 1][j] + 1, D[i - 1][j - 1] + 1 - int(str_1[i - 1] == str_2[j - 1]))
    return D[-1][-1]


letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'а', 'в', 'с', 'е', 'н', 'к', 'м', 'о', 'р', 'т', 'х', 'у']


# Функция для чтения изображения из json файла
def read_img_from_json(path):
    filelist = os.listdir(path)
    all_img, number = [], []
    for item in filelist:
        if item.endswith(('json')):
            all_img.append(item)
            src = os.path.join(os.path.abspath(path), item)
            file = open(src)
            f = json.load(file)['shapes']
            number.append(''.join([i['label'] for i in f]))
    return all_img, number

def convert_img(path,img_name):
    src = os.path.join(os.path.abspath(path), img_name)
    file = open(src)
    return img_b64_to_arr(json.load(file)['imageData'])

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


tich_file, number = read_img_from_json(img_path)
elapsed_time, result = [], {}

interpreter = tf.lite.Interpreter(model_path=model_resnet_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


for i in tqdm(range(len(tich_file))):
    image0 = convert_img(img_path, tich_file[i])
    true_number = number[i]
    image_height, image_width, _ = image0.shape
    image = cv2.resize(image0, (1024, 1024))
    image = image.astype(np.float32)
    X_data1 = np.float32(image.reshape(1, 1024, 1024, 3))
    start_time1 = datetime.now()
    interpreter.set_tensor(input_details[0]['index'], X_data1)
    interpreter.invoke()
    end_time1=datetime.now()
    detection = interpreter.get_tensor(output_details[0]['index'])
    net_out_value2 = interpreter.get_tensor(output_details[1]['index'])
    net_out_value3 = interpreter.get_tensor(output_details[2]['index'])
    net_out_value4 = interpreter.get_tensor(output_details[3]['index'])
    img = image0
    razmer = img.shape

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts from one colour space to the other
    img3 = img[:, :, :]

    box_x = int(detection[0, 0, 0] * image_height)
    box_y = int(detection[0, 0, 1] * image_width)
    box_width = int(detection[0, 0, 2] * image_height)
    box_height = int(detection[0, 0, 3] * image_width)
    if np.min(detection[0, 0, :]) >= 0:
        image = img3[box_x:box_width, box_y:box_height, :]
        grayscale = rgb2gray(image)
        edges = canny(grayscale, sigma=3.0)
        if HOUGH_TRANSFORM:
            out, angles, distances = hough_line(edges)
            _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
            angle = np.mean(np.rad2deg(angles_peaks))
            if 0 <= angle <= 90:
                rot_angle = angle - 90
            elif -45 <= angle < 0:
                rot_angle = angle - 90
            elif -90 <= angle < -45:
                rot_angle = 90 + angle
            if abs(rot_angle) > 20:
                rot_angle = 0
            rotated = rotate(image, rot_angle, resize=True) * 255
            rotated = rotated.astype(np.uint8)
            rotated1 = rotated[:, :, :]
            minus = np.abs(int(np.sin(np.radians(rot_angle)) * rotated.shape[0]))
            if rotated.shape[1] / rotated.shape[0] < 2 and minus > 10:
                rotated1 = rotated[minus:-minus, :, :]
            lab = cv2.cvtColor(rotated1, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            img = final  # лучше работает при плохом освещении
        else:
            img = image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (NUMBER_WIDTH, NUMBER_HEIGHT))
        img = img.astype(np.float32)
        img /= 255

        img1 = img.T
        X_data1 = np.float32(img1.reshape(1, NUMBER_WIDTH, NUMBER_HEIGHT, 1))
        interpreter1 = tf.lite.Interpreter(model_path=model_nomer_paths)
        interpreter1.allocate_tensors()
        input_details1 = interpreter1.get_input_details()
        output_details1 = interpreter1.get_output_details()
        start_time2 = datetime.now()
        interpreter1.set_tensor(input_details1[0]['index'], X_data1)
        interpreter1.invoke()
        net_out_value = interpreter1.get_tensor(output_details1[0]['index'])
        end_time2 = datetime.now()
        pred_text = decode_batch(net_out_value)
        result[true_number] = pred_text
    else:
        pred_texts = ''
        result[true_number] = pred_text
    elapsed_time.append((end_time1 - start_time1 - (end_time2-start_time2)).total_seconds())

# Расчет необходимых метрик
latency = np.mean(elapsed_time)
lev_distance = [levenstein(i[0], i[1][0]) for i in result.items()]
true_positive = sum([len(i[0]) - levenstein(i[0], i[1][0]) for i in result.items()])
precision = true_positive / sum([len(i[0]) for i in result.values()])
recall = true_positive / sum([len(i) for i in result.keys()])
ocr_total = len(number)
ocr_error = sum([ int(levenstein(i[0], i[1][0]) >= LEVENSTEIN_TRESHOLD_ERROR) for i in result.items()])
ocr_missed = sum([ int(levenstein(i[0], i[1][0]) == LEVENSTEIN_TRESHOLD_MISSED) for i in result.items()])
ocr_miss_rate = ocr_missed / ocr_total
ocr_normal = 1 - (ocr_missed + ocr_error) / ocr_total
metrics = {'recall': recall,
           'precision': precision,
           'latency': latency,
           'ocr_total': ocr_total,
           'ocr_error': ocr_error,
           'ocr_missed': ocr_missed,
           'ocr_miss_rate': ocr_miss_rate,
           'ocr_normal': ocr_normal}
jsonString = json.dumps(metrics, indent=0)
with open(args.metrics + '/metrics.json', 'w') as f:
    f.write(jsonString)
