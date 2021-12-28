import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from numpy import random

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import onnxruntime as rt

# from models.models import Darknet

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def load_image_from_directory(directory:str)->list:
    imageList = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            imagePath = os.path.join(directory, filename)
            image = cv2.imread(imagePath)
            imageList.append(image)
        else:
            continue
    return imageList

class YOLOR(object):
    def __init__(self,
            imgsz = (416, 416), 
            pipe_weights = './onnx/deepSeaPipe499.onnx',  
            pipe_names = 'data/pipe.names',
            etc_weights = './onnx/deepSeaEtc.onnx',
            etc_names='data/etc.names'):
        self.pipe_names = load_classes(pipe_names)
        self.pipe_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.pipe_names))]
        self.pipe_model = rt.InferenceSession(pipe_weights)

        self.etc_names = load_classes(etc_names)
        self.etc_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.etc_names))]
        self.etc_model = rt.InferenceSession(etc_weights)

        self.imgsz = imgsz
        # Load model
        

    def detect(self, bgr_img, threshold = 0.2):   
        # Prediction
        ## Padded resize
        inp = letterbox(bgr_img, new_shape=self.imgsz, auto_size=64,auto = False,scaleFill=True)[0]
        # print(inp.shape)
        # inp = cv2.resize(bgr_img, self.imgsz)
        inp = inp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp, 0)
        # inp = np.transpose(inp, (0, 3, 1, 2))
        print(inp.shape)
        ## Convert to torch
        
        
        ## Inference Pipe
        t1 = time.time()
        pipe_inputs = {self.pipe_model.get_inputs()[0].name: inp}
        pipe_pred = self.pipe_model.run(None, pipe_inputs)[0]
        t2 = time.time()
        ## Inference ETC
        td1 = time.time()
        etc_inputs = {self.etc_model.get_inputs()[0].name: inp}
        etc_pred = self.etc_model.run(None, etc_inputs)[0]
        td2 = time.time()
        ## Apply NMS
        with torch.no_grad():
            pipe_pred = non_max_suppression(torch.tensor(pipe_pred), conf_thres=threshold, iou_thres=0.6)
            etc_pred = non_max_suppression(torch.tensor(etc_pred), conf_thres=threshold, iou_thres=0.6)
        t3 = time.time()
        print('Inference Pipe: {}'.format(t2-t1))
        print('Inference Etc: {}'.format(td2-td1))
        print('NMS: {}'.format(t3-t1))
    
        # Process detections
        visualize_img = bgr_img.copy()
        pipe_det = pipe_pred[0]  # detections per image
        etc_det = etc_pred[0]  # detections per image
        if pipe_det is not None and len(pipe_det):
            # Rescale boxes from img_size to im0 size
            _, _, height, width = inp.shape
            h, w, _ = bgr_img.shape
            pipe_det[:, 0] *= w/width
            pipe_det[:, 1] *= h/height
            pipe_det[:, 2] *= w/width
            pipe_det[:, 3] *= h/height
            for x1, y1, x2, y2, conf, cls in pipe_det:       # x1, y1, x2, y2 in pixel format
                label = '%s %.2f' % (self.pipe_names[int(cls)], conf)
                plot_one_box((x1, y1, x2, y2), visualize_img, label=label, color=self.pipe_colors[int(cls)], line_thickness=3)
        if etc_det is not None and len(etc_det):
            # Rescale boxes from img_size to im0 size
            _, _, height, width = inp.shape
            h, w, _ = bgr_img.shape
            etc_det[:, 0] *= w/width
            etc_det[:, 1] *= h/height
            etc_det[:, 2] *= w/width
            etc_det[:, 3] *= h/height
            for x1, y1, x2, y2, conf, cls in etc_det:       # x1, y1, x2, y2 in pixel format
                label = '%s %.2f' % (self.etc_names[int(cls)], conf)
                plot_one_box((x1, y1, x2, y2), visualize_img, label=label, color=self.etc_colors[int(cls)], line_thickness=3)

        #cv2.imwrite('result/deep_sea/result.jpg', visualize_img)
        return visualize_img

if __name__ == '__main__':
    model = YOLOR()
    #img = cv2.imread('test_images/deep_sea/20201107122805838_png_jpg.rf.83bf790272b637c2d2428015ef0e8f8b.jpg')
    img_list = load_image_from_directory("test_images/deep_sea")
    result_number = 0
    for img in img_list:
        t0 = time.time()
        detected_img = model.detect(img)
        cv2.imwrite(f'result/deep_sea_ensemble/result+{result_number}.jpg', detected_img)
        result_number += 1
        t1 = time.time()
        print(f"ProcessTime= {t1-t0} sec")
