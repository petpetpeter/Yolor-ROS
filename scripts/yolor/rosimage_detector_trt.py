#!/usr/bin/env python3

import numpy as np
import os
import math
import time
import cv2
import struct
from numpy.core.numeric import NaN
import open3d as o3d
import matplotlib.pyplot as plt
import glob
import ctypes
import warnings


#ROS Required
import rospy
import ros_numpy
from std_msgs.msg import Header,String
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, Image,CompressedImage
from std_msgs.msg import Bool,Float32MultiArray
from visualization_msgs.msg import Marker,MarkerArray
from tomato_detection.srv import SelectTomato,SelectTomatoResponse
from geometry_msgs.msg import Point,Pose,PoseStamped,PoseArray,PointStamped


#Yolor & Tensorrt
import torch
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from exec_backends.trt_loader import TrtModel


class YOLOR(object):
    def __init__(self,conf=0.7):
        max_size = 896
        ENGINE_PATH     = os.path.dirname(os.path.abspath(__file__))+'/yolor_p6_fp16.trt'
        CLASS_NAME_PATH = os.path.dirname(os.path.abspath(__file__))+'/data/coco.names'
        self.names = self.load_classes(CLASS_NAME_PATH)
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = (max_size, max_size)
        # Load model
        self.conf = conf
        self.model = TrtModel(ENGINE_PATH, max_size,total_classes = len(self.names))
        self.bboxDict = dict((name,[]) for name in self.names)
        print(self.bboxDict)

    def load_classes(self,path):
    # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)

    def resetDict(self,bboxDict):
        for k in bboxDict.keys():
            bboxDict[k] = []

    def detect(self, bgr_img):   
        self.resetDict(self.bboxDict)
        # Prediction
        ## Padded resize
        inp = letterbox(bgr_img, new_shape=self.imgsz, auto_size=64)[0]
        inp = inp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp, 0)
        #print(inp.shape,type(inp))        
        
        ## Inference
        t1 = time.time()
        pred = self.model.run(inp)[0]
        t2 = time.time()
        ## Apply NMS
        with torch.no_grad():
            pred = non_max_suppression(torch.tensor(pred), conf_thres=self.conf, iou_thres=0.6)
        t3 = time.time()
        print('Inference: {}'.format(t2-t1))
        print('NMS: {}'.format(t3-t2))
        #print('FPS: ', 1/(t3-t1))
    
        # Process detections
        visualize_img = bgr_img.copy()
        det = pred[0]  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            _, _, height, width = inp.shape
            h, w, _ = bgr_img.shape
            det[:, 0] *= w/width
            det[:, 1] *= h/height
            det[:, 2] *= w/width
            det[:, 3] *= h/height
            for x1, y1, x2, y2, conf, cls in det:       # x1, y1, x2, y2 in pixel format
                label_conf = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box((x1, y1, x2, y2), visualize_img, label=label_conf, color=self.colors[int(cls)], line_thickness=3)
                label = self.names[int(cls)]
                self.bboxDict[label].append((x1.item(), y1.item(), x2.item(), y2.item()))
        return visualize_img,self.bboxDict


class DepthImageHandler(object):
    def __init__(self):
        self.CAMINFO = {'topic': '/camera/color/camera_info', 'msg': CameraInfo}
        self.COLOR = {'topic': '/camera/color/image_raw', 'msg': Image}
        self.DEPTH = {'topic': '/camera/depth/image_raw', 'msg': Image}
        self.isCamInfo = False
        self.PC = {'topic': '/hand_tomatoPC', 'msg': PointCloud2}
        
        self.H = 720
        self.W = 1280
        self.header = Header() #Use for point cloud publisher

        self.color_image = np.empty((self.H, self.W ,3), dtype=np.uint8)
        self.depth_image = np.empty((self.H, self.W), dtype=np.uint16)
        self.aligned_image  = np.empty((self.H, self.W), dtype=np.uint8)
        self.mask           = np.empty((self.H, self.W), dtype=np.bool)
        self.mask_image     = np.empty((self.H, self.W), dtype=np.uint8)
        self.mask_depth     = np.empty((self.H, self.W), dtype=np.uint8)

        self.markerColors = np.random.uniform(0,1,[80,3])

        self.camera_matrix = np.array([[0.0, 0, 0.0], [0, 0.0, 0.0], [0, 0, 1]], dtype=np.float32) 
        self.camera_distortion = np.array([0,0,0,0,0], dtype=np.float32)
        self.model = YOLOR(conf=0.9)

        self.SCORE_THR = 0.85 #  cv2.getTrackbarPos('score','image')
        
        #create trackbar
        cv2.namedWindow('score_bar')
        cv2.createTrackbar('score','score_bar',1,100,self.nothing)
        cv2.setTrackbarPos('score','score_bar', int (self.SCORE_THR * 100) )

    def nothing(self):
        pass #for trackbar

    def camInfoCallback(self, msg):
        self.header = msg.header
        self.K = msg.K
        self.width = msg.width  
        self.height = msg.height
        self.ppx = msg.K[2]
        self.ppy = msg.K[5]
        self.fx = msg.K[0]
        self.fy = msg.K[4] 
        
        self.cam_distortion_model = msg.distortion_model
        self.k1 = msg.D[0]
        self.k2 = msg.D[1]
        self.t1 = msg.D[2]
        self.t2 = msg.D[3]
        self.k3 = msg.D[4]
        self.isCamInfo = True
        self.camera_matrix = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]], dtype=np.float32) 
        self.camera_distortion = np.array([self.k1,self.k2,self.t1,self.t2,self.k3], dtype=np.float32)

    def colorCallback(self, msg):
        self.color_image = ros_numpy.numpify(msg)
        #self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)

    def depthCallback(self, msg):
        #self.depth_image = ros_numpy.numpify(msg).astype(np.uint16) # //RealImage
        numpyImage = ros_numpy.numpify(msg)
        numpyImage = np.nan_to_num(numpyImage, copy=True, nan=0.0)
        self.depth_image = numpyImage
    
    def publishPoint3(self,pos3):
        point = PointStamped()
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = self.header.frame_id

        point.point.x = pos3[0]
        point.point.y = pos3[1]
        point.point.z = pos3[2]
        
        self.point_pub.publish(point)
    


    def publishObjectPos3MarkerArray(self,camPosDict: dict)-> None:
        objectMarkerArray = MarkerArray()
        colors = self.markerColors
        id = 0
        numClass = 0 
        for name in camPosDict:
            for pos3 in camPosDict[name]:
                objectMarker = Marker()
                objectMarker.color.r = colors[numClass][0]
                objectMarker.color.g = colors[numClass][1]
                objectMarker.color.b = colors[numClass][2]
                objectMarker.color.a = 1.0
                objectMarker.header.frame_id = self.header.frame_id # Camera Optical Frame
                objectMarker.header.stamp = rospy.Time.now()
                objectMarker.type = 2 # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
                # Set the scale of the marker
                objectMarker.scale.x = 0.1
                objectMarker.scale.y = 0.1
                objectMarker.scale.z = 0.1
                # Set the color
                objectMarker.id = id
                objectMarker.pose.position.x = pos3[0]
                objectMarker.pose.position.y = pos3[1]
                objectMarker.pose.position.z = pos3[2]
                objectMarker.lifetime = rospy.Duration(0.1)
                objectMarkerArray.markers.append(objectMarker)
                id += 1
            numClass += 1
        self.array_pub.publish(objectMarkerArray)
    
    def publishImage(self,image):
        msg = ros_numpy.msgify(Image, image,encoding = "bgr8")
        # Publish new image
        self.image_pub.publish(msg)
    

    def pixel_crop(self,img, dim,pixel):
        width, height = img.shape[1], img.shape[0]
        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(pixel[0]), int(pixel[1])
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
        
    def depthPixelToPoint3(self, depth_image, U, V):
        V =  np.clip(V,0,self.height-1)
        U =  np.clip(U,0,self.width-1)  
        
        x = (U - self.K[2])/self.K[0]
        y = (V - self.K[5])/self.K[4]     
        nearPixel = self.pixel_crop(depth_image,[5,5],(U,V))
        meanDepth = np.mean(nearPixel[np.nonzero(nearPixel)]) #Mean of non zero in neighborhood pixel
        if np.isnan(meanDepth):
            rospy.logwarn("Nan depth value, 0 mean depth is returned")
            meanDepth = 0
        #print(meanDepth,"mean")
        z = meanDepth # /1000 for real camera
        x *= z
        y *= z
        # print(U,V,x,y,z)
        point3 = [x, y, z]
        return point3
    
    def pos3FromBboxes(self, depthImage, bboxes:list)->list:
        pos3List = []
        for (x1,y1,x2,y2) in bboxes:
            cX,cY = (int((x2+x1)/2),int((y2+y1)/2))
            pos3List.append(self.depthPixelToPoint3(depthImage,cX,cY))
        return pos3List

    def process(self):
        #t1 = time.time()
        color_image, depth_image    = self.color_image, self.depth_image
        detected_image,bboxDict = self.model.detect(color_image)
        #print(bboxDict)
        camPosDict = dict((name,self.pos3FromBboxes(depth_image,bboxDict[name])) for name in bboxDict if len(bboxDict[name]) > 0)
        print(camPosDict)
        self.publishObjectPos3MarkerArray(camPosDict)
        self.publishImage(detected_image)
        cv2.imshow('frame', detected_image)
        cv2.waitKey(1)

    def rosinit(self):
        rospy.init_node('markerFinder', anonymous=True)
        #r = rospy.Rate(60)
        rospy.Subscriber(self.CAMINFO['topic'], self.CAMINFO['msg'], self.camInfoCallback)
        rospy.Subscriber(self.COLOR['topic'], self.COLOR['msg'], self.colorCallback)
        rospy.Subscriber(self.DEPTH['topic'], self.DEPTH['msg'], self.depthCallback)
        #rospy.Subscriber(self.PC['topic'],    self.PC['msg'],    self.pcCallback)
    
        ###publisher        
        self.array_pub = rospy.Publisher('/objectsMarkers', MarkerArray, queue_size=1)
        self.image_pub = rospy.Publisher('/detectedImage', Image, queue_size=1)
                    
        while not rospy.is_shutdown():
                if self.isCamInfo: #Wait for camera ready
                    self.process()

if __name__ == '__main__':
    try:
        _depthImageHandler = DepthImageHandler()
        _depthImageHandler.rosinit()

    except rospy.ROSInterruptException:
        pass
