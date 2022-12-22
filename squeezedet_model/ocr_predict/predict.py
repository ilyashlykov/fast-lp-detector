'''
Показана работа с моделями распознавания пластинки номера и символов номера загружаемых в tensorflow

Необходимо установить следующие пакеты:
pip install numpy
pip install tensorflow
pip install opencv-python

'''

import cv2
import numpy as np
from model import Model

if __name__ == "__main__":

    plate_model = Model('plate_model.pb', 'plate_model.json')
    chars_model = Model('chars_model.pb', 'chars_model.json')

    image = cv2.imread('sample.jpg')
    image_shape = np.shape(image)
    image = cv2.resize(image, (image_shape[1] * 10, image_shape[0] * 10))

    plate_detections = plate_model.predict(image)
    char_detections = chars_model.predict(image)

    for bbox, conf, class_name in zip(plate_detections[0], plate_detections[1], plate_detections[3]):
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),  (0, 255, 0) , 2)
        cv2.putText(image, class_name, (int(bbox[0]), int(bbox[3] + 20 )), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, '{:.3f}'.format(conf) , (int(bbox[0]), int(bbox[3] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    for bbox, conf, class_name in zip(char_detections[0], char_detections[1], char_detections[3]):
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),  (0, 255, 220) , 2)
        cv2.putText(image, class_name, (int(bbox[0]), int(bbox[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 220), 2, cv2.LINE_AA)
        cv2.putText(image, '{:.3f}'.format(conf) , (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 220), 2, cv2.LINE_AA)
    cv2.imshow("image", image)
    cv2.waitKey(0)
