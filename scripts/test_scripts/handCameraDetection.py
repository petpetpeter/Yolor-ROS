#!/usr/bin/env python

import numpy as np
import os
import math
import time
import cv2
import struct
import open3d as o3d
import matplotlib.pyplot as plt
import glob
import ctypes


#ROS Required
import rospy
import ros_numpy
from std_msgs.msg import Header,String
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, Image
from std_msgs.msg import Bool,Float32MultiArray
from visualization_msgs.msg import Marker,MarkerArray
from realsense2_camera.srv import SelectTomato,SelectTomatoResponse
from geometry_msgs.msg import Point,Pose,PoseStamped,PoseArray,PointStamped

#MMDection Required
import json
import mmcv
import torch
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO as coco
from mmdet.apis import inference_detector, init_detector





class DepthImageHandler(object):
    def __init__(self):
        self.CAMINFO = {'topic': '/camera/color/camera_info', 'msg': CameraInfo}
        self.COLOR = {'topic': '/camera/color/image_raw', 'msg': Image}
        self.DEPTH = {'topic': '/camera/aligned_depth_to_color/image_raw', 'msg': Image}
        self.isCamInfo = False
        self.PC = {'topic': '/hand_tomatoPC', 'msg': PointCloud2}
        
        self.H = 720
        self.W = 1280
        self.cropSize = 1 #1 for no crop
        self.header = Header() #Use for point cloud publisher
        self.fields = [PointField('x', 0, 7, 1), PointField('y', 4, 7, 1), PointField('z', 8, 7, 1), PointField('rgb', 16, 7, 1)]
        self.points = []
        self.pc = PointCloud2()

        self.color_image = np.empty((self.H, self.W ,3), dtype=np.uint8)
        self.depth_image = np.empty((self.H, self.W), dtype=np.uint16)
        self.aligned_image  = np.empty((self.H, self.W), dtype=np.uint8)
        self.mask           = np.empty((self.H, self.W), dtype=np.bool)
        self.mask_image     = np.empty((self.H, self.W), dtype=np.uint8)
        self.mask_depth     = np.empty((self.H, self.W), dtype=np.uint8)
        self.cropPointCloud = o3d.geometry.PointCloud() 

        self.camera_matrix = np.array([[0.0, 0, 0.0], [0, 0.0, 0.0], [0, 0, 1]], dtype=np.float32) 
        self.camera_distortion = np.array([0,0,0,0,0], dtype=np.float32)

        self.pickUV = [self.H//2,self.W//2]


        PATH_TO_PROJECT     = os.path.dirname(os.path.abspath(__file__))+'/../mmdetection/tomato_3cls'
        ANNOTATION_FILENAME = 'trainval.json'
        CHECKPOINT_FILENAME = 'epoch_1000.pth'
        CONFIG_FILENAME     = 'tomato_3cls_htc.py'

        # PATH_TO_PROJECT     = os.path.dirname(os.path.abspath(__file__))+'/mmdetection/yolact_tomato3cls'
        # ANNOTATION_FILENAME = 'trainval.json'
        # CHECKPOINT_FILENAME = 'epoch_1.pth'
        # CONFIG_FILENAME     = 'yolact_r50_1x8_coco_tomato.py'


        annotation_file     = os.path.join(PATH_TO_PROJECT, ANNOTATION_FILENAME)
        json_file           = open(annotation_file)
        coco                = json.load(json_file)
        checkpoint_file     = os.path.join(PATH_TO_PROJECT, CHECKPOINT_FILENAME)
        config_file         = os.path.join(PATH_TO_PROJECT, CONFIG_FILENAME)
        self.class_names    = [category['name'] for category in coco['categories']] 
        # print(self.class_names)
        self.modelDet          = init_detector(config_file, checkpoint_file, device=torch.device('cuda', 0))
        self.class_names.append('reserved1')
        self.modelDet.CLASSES  = self.class_names
        self.SCORE_THR = 0.85 #  cv2.getTrackbarPos('score','image')
        
        #create trackbar
        cv2.namedWindow('score_bar')
        cv2.createTrackbar('score','score_bar',1,100,self.nothing)
        cv2.setTrackbarPos('score','score_bar', int (self.SCORE_THR * 100) )

        self.realTimeDetectionImage = np.zeros([self.H, self.W, 3], dtype=np.float)
        self.fullRedTomato = self.Tomato(stage = "b_fully_ripened")
        self.halfRedTomato = self.Tomato(stage = "b_half_ripened")
        self.greenTomato = self.Tomato(stage = "b_green")
        self.getLabelInstanceDict = {0:self.fullRedTomato ,
                                    1:self.halfRedTomato,
                                    2:self.greenTomato}


    class Tomato():
        def __init__(self,stage):
            self.stage = stage
            self.number = 0
            self.bboxes = []
            self.masks = []
            self.camPos = []
        def __str__(self):
            return f"tomato class : {self.stage}\nnumber :{self.number}"
        def getClosetTomatoPos(self,targetPos):
            Distances = [math.sqrt(sum((tomatoPos - targetPos) ** 2.0 for tomatoPos, targetPos in zip(tomatoPos, targetPos))) for tomatoPos in self.camPos]
            min_index = Distances.index(min(Distances))
            return self.camPos[min_index]
        def reset(self):
            self.number = 0
            self.bboxes = []
            self.masks = []
            self.camPos = []

    def nothing(self):
        pass


    def camInfoCallback(self, msg):
        self.header = msg.header
        self.K = msg.K
        self.width = msg.width  
        self.height = msg.height
        self.ppx = msg.K[2]
        self.ppy = msg.K[5]
        self.fx = msg.K[0]
        self.fy = msg.K[4] 
        
        self.model = msg.distortion_model
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
        self.depth_image = ros_numpy.numpify(msg).astype(np.uint16)
    
    def pcCallback(self, msg):
        self.fields = msg.fields
        #print(self.fields)

    

    
    
    def find_nearest_nonzero(self,depth_img, target):
        #nonzero = cv2.findNonZero(depth_img)
        #distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
        #nearest_index = np.argmin(distances)
        zeroPixel = depth_img[target[1],target[0]]
        count = 0
        while zeroPixel == 0:
            count += 1
            zeroPixel = depth_img[target[1],target[0]+count]
            if count > 5:
                break

        return target[0]+count,target[1]

    def depthToPoint3(self, depth_image, U, V):
        V =  np.clip(V,0,self.height-1)
        U =  np.clip(U,0,self.width-1)  
        
        x = (U - self.K[2]/self.cropSize)/self.K[0]
        y = (V - self.K[5]/self.cropSize)/self.K[4]     
        nearPixel = self.pixel_crop(depth_image,[5,5],(U,V))
        # imgplot = plt.imshow(np.nonzero(nearPixel))
        # plt.show()
        meanDepth = np.mean(nearPixel[np.nonzero(nearPixel)])
        #print(meanDepth,"mean")
        z = meanDepth / 1000
        x *= z
        y *= z
        # print(U,V,x,y,z)
        point3 = [x, y, z]
        return point3
        # Checks if a matrix is a valid rotation matrix.

    def pointToDepthUV(self, point3):
        [x,y,z] = point3  
        
        u = (x*self.K[0]/z) + self.K[2]/self.cropSize
        v = (y*self.K[4]/z) + self.K[5]/self.cropSize   
        return [int(u),int(v)]

    
    def rgbToFloat(self, rgb):
        return struct.unpack('f', struct.pack('i', rgb))[0]


    def depthToPoints(self, depth_image,color_image):
        [height, width] = depth_image.shape
        #print(depth_image.shape,color_image.shape)
        #print(self.K)
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        #print(u)
        x = (u.flatten() - self.K[2]/self.cropSize)/self.K[0]
        y = (v.flatten() - self.K[5]/self.cropSize)/self.K[4]

        z = depth_image.flatten() /1000
        

        color_points = np.reshape(color_image, [-1,3])
        color_points = color_points[np.nonzero(z)]
        #rgb_int = [self.rgb2hex(q,w,e) for q,w,e in color_points[:]]
        
        #print(rgb_int)
        #bgr --> rgb
        r = color_points[:,2]
        g = color_points[:,1]
        b = color_points[:,0]
        rgb_int = 65536 * r   + 256 * g  + b
        rgb_int = [self.rgbToFloat(r) for r in rgb_int]
        
        #print(rgb_int)
        x = np.multiply(x,z)
        y = np.multiply(y,z)
        #print(np.nonzero(z))
        
        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]
        #print(z)
        r = r[np.nonzero(z)]
        g = g[np.nonzero(z)]
        b = b[np.nonzero(z)]
        #rgb_int = rgb_int[np.nonzero(z)]
        #rgb_int = [self.rgb2hex(x,y,z) for x in r for y in g for z in b]
        #print(len(x),len(y),len(z),len(r),len(g),len(b),len(rgb_int))
        points = np.stack((x,y,z,rgb_int), axis = -1) 
        #print(points)
        return points

    def isRotationMatrix(self,R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

   

    def boardCastMarkerFromCameraTF(self,markerPose):
        markerMessage = Float32MultiArray()
        markerMessage.data = markerPose.tolist()
        self.markerXYZRPY_pub.publish(markerMessage)
        #print(markerMessage)

    def findRotationFromTo(self,VecA,VecB):
        v = np.cross(VecA,VecB)
        s = np.linalg.norm(v)
        c = np.dot(VecA,VecB)
        I = np.identity(3)
        vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
        k = np.matrix(vXStr)
        r = I + k + np.matmul(k,k) * ((1 -c)/(s**2))
        return r

    def publishPoint3(self,pos3):
        point = PointStamped()
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = self.header.frame_id


        point.point.x = pos3[0]
        point.point.y = pos3[1]
        point.point.z = pos3[2]
        
        self.point_pub.publish(point)
    
    def publishPointCloud(self, pc_pub, points):
        pc = point_cloud2.create_cloud(self.header, self.fields, points)
        self.cropPointCloud = pc
        pc_pub.publish(pc)
    
    def publishVec3Marker(self, start, end):
        marker = Marker()

    def getColorBound(self):
    
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")

        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")
        
        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])
        
        # l_b = np.array([49,100,33])
        # u_b = np.array([147,255,255])
        return l_b,u_b


    def getMask(self,color,img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #imgplot = plt.imshow(hsv)
        #plt.show()
        if color == "all":
            l_b = np.array([0, 10, 0])
            u_b = np.array([50, 255, 150])
        if color == "red":
            l_b = np.array([0,180,60])
            u_b = np.array([10,255,255])
        if color == "green":
            l_b = np.array([10, 10, 0])
            u_b = np.array([255, 255, 150])

        mask_img = cv2.inRange(hsv, l_b, u_b)
        return mask_img

    def centroid(self,img):
        M   = cv2.moments(img)
        if  M["m00"] == 0 or M["m00"] == 0 :
            return
        
        cX  = int(M["m10"] / M["m00"])
        cY  = int(M["m01"] / M["m00"])
        return cX, cY 

    def center_crop(self,img, dim):
        width, height = img.shape[1], img.shape[0]
        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
    
    def pixel_crop(self,img, dim,pixel):
        width, height = img.shape[1], img.shape[0]
        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(pixel[0]), int(pixel[1])
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
    
    def sphereFit(self,spX,spY,spZ):
        #   Assemble the A matrix
        
        spX = np.array(spX)
        spY = np.array(spY)
        spZ = np.array(spZ)
        A = np.zeros((len(spX),4))
        A[:,0] = spX*2
        A[:,1] = spY*2
        A[:,2] = spZ*2
        A[:,3] = 1

        #   Assemble the f matrix
        f = np.zeros((len(spX),1))
        f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
        C, residules, rank, singval = np.linalg.lstsq(A,f)

        #   solve for the radius
        t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
        radius = math.sqrt(t)
            #print(radius, C[0][0], C[1][0], C[2][0])

        return [radius, C[0][0], C[1][0], C[2][0]]

    def tomatoDetect(self,color_image,depth_image,isUpdateValue = True):
        
        if isUpdateValue:
            self.fullRedTomato.reset()
            self.halfRedTomato.reset()
            self.greenTomato.reset()

        start_time = time.time()
        result = inference_detector(self.modelDet, color_image)
        execute_time = (time.time() - start_time)
        #print("FPS: ", 1.0 / execute_time, "Detection time: ", (execute_time)*1000 , " ms")
        assert isinstance(self.class_names, (tuple, list)) #Check type of class name
        wait_time=1
        show=True
        score_thr= r = cv2.getTrackbarPos('score','score_bar') / 100.0 #self.SCORE_THR
        out_file=None

        
        ##result type Check
        if self.modelDet.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']

        resulted_image = mmcv.imread(color_image)
        resulted_image = resulted_image.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        ##result type Check
        bboxes = np.vstack(bbox_result)
        
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        
        labels = np.concatenate(labels)


        #print(labels.shape)
        #print(bboxes.shape) #(n,5) lastest is score
        
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0] #List of detected Object
            np.random.seed(42)

            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)

                mask = segms[i]
                bbox = bboxes[i]

                # for centroid calculation modified
                mask_int = mask.astype(np.uint8)*255
                if self.centroid(mask_int) is None :
                    break
                cX, cY = self.centroid(mask_int)
                camPos = self.depthToPoint3(depth_image,cX,cY)

                if isUpdateValue:
                    tomatoInstance = self.getLabelInstanceDict[labels[i]]    
                    tomatoInstance.masks.append(mask.astype(np.uint8)*255)
                    tomatoInstance.bboxes.append(bbox)
                    tomatoInstance.camPos.append(camPos)
                    tomatoInstance.number = len(tomatoInstance.masks)
                    

                
                color_mask = color_masks[labels[i]]
                cv2.circle(resulted_image, (cX, cY), 5, (0, 0, 0), -1)
                resulted_image[mask] = resulted_image[mask] * 0.6 + color_mask * 0.4
        
        # draw bounding boxes
        bbox_color='blue'
        text_color='blue'
        thickness=1
        font_scale=0.5
        win_name='Hybrid Task Cascade'
        show=False
        wait_time=1

        resulted_image = mmcv.imshow_det_bboxes(
            resulted_image,
            bboxes,
            labels,
            class_names=self.modelDet.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        self.realTimeDetectionImage = resulted_image

        #print(self.fullRedTomato)

        return resulted_image

    def findTomatoCallBack(self,req):
        print(req.gripperPos,"geipperPos")
        gripperPos_cam = req.gripperPos
        color_image, depth_image    = self.color_image, self.depth_image
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        #depth_image[depth_image > 800] = 0
        
        detectedImage = self.tomatoDetect(color_image,depth_image)

        # visual_color_image = np.copy(color_image)
        # tomatoMask,tomatoC = self.getMask("red",color_image)
        # # coreMask,coreC = self.getMask("green",color_image)
        #print(self.fullRedTomato.number)
        if self.fullRedTomato.number == 0:
            return SelectTomatoResponse([0,0,0])

        tomatoMask = self.fullRedTomato.masks[0]# if self.fullRedTomato.number > 0 else np.ones([self.H, self.W, 1], dtype=np.uint8)
        masked_tomato = np.zeros_like(color_image)
        masked_tomato[np.nonzero(tomatoMask)] = color_image[np.nonzero(tomatoMask)]
        redTomatoMask = self.getMask("red",masked_tomato)
        masked_redTomato = np.zeros_like(color_image)
        masked_redTomato[np.nonzero(redTomatoMask)] = color_image[np.nonzero(redTomatoMask)]
        
        masked_depth = np.zeros_like(depth_image)
        masked_depth[np.nonzero(redTomatoMask)] = depth_image[np.nonzero(redTomatoMask)]
        tomatoPoints = self.depthToPoints(masked_depth,color_image)
        self.publishPointCloud(self.tomatoPC_pub,tomatoPoints)

        tomatoPos3 = self.fullRedTomato.getClosetTomatoPos(gripperPos_cam)
        print(tomatoPos3,"tomato")
        self.publishPoint3(tomatoPos3)

        

        # imgplot = plt.imshow(masked_depth)
        # plt.show()

        
        
        # tomatoChannel = np.moveaxis(tomatoPoints, 0, 1)
        # tomatoX = tomatoChannel[0]
        # tomatoY = tomatoChannel[1]
        # tomatoZ = tomatoChannel[2]
        # sphere = self.sphereFit(tomatoX,tomatoY,tomatoZ)
        
        # tomatoCenter = [sphere[1],sphere[2],sphere[3]] #rxyz
        # tomatoSuck = [sphere[1]-sphere[0],sphere[2],sphere[3]]
        #print(tomatoCenter,"tomato")

        
        
        
        # if sum(tomatoSuck) != 0:
        #     suckImagePos = self.pointToDepthUV(tomatoSuck)
        # else:
        #     return SelectTomatoResponse(tomatoSuck)

        # collisionArray = [0,0,0]
        # self.suckUV = suckImagePos

        # suctionCheckBlock = self.pixel_crop(tomatoMask,[50,50],self.suckUV)
        # suctionBool =  1 if (np.count_nonzero(suctionCheckBlock))/2500 < 0.95 else 0
        # collisionArray = [0,suctionBool,1]
        # print(np.count_nonzero(suctionCheckBlock))
        # print(suctionBool,"suctionBOol")
        #print(collisionCheckBlock)

        return SelectTomatoResponse(tomatoPos3)
    
    def collisionCheckingCallBack(self,req):
        pickPosCam = req.gripperPos
        #self.publishPoint3(pickPosCam)
        self.pickUV = self.pointToDepthUV(pickPosCam)
        print(self.fullRedTomato)
        color_image, depth_image    = self.color_image, self.depth_image
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        depth_image[depth_image > 800] = 0
        
        # waitTIme = time.time() + 2
        # foundTomato = 0
        # while time.time() < waitTIme:
        #     detectedImage = self.tomatoDetect(color_image)
        #     if self.fullRedTomato.number > 0:
        #         foundTomato = 1
        #         break
        # if not foundTomato:
        #     return SelectTomatoResponse([0,0,0])


        tomatoMask = self.fullRedTomato.masks[0] if self.fullRedTomato.number > 0 else np.ones([self.H, self.W, 1], dtype=np.uint8)
        masked_tomato = np.zeros_like(color_image)
        masked_tomato[np.nonzero(tomatoMask)] = color_image[np.nonzero(tomatoMask)]
        redTomatoMask = self.getMask("red",masked_tomato)
        suctionCheckBox= self.pixel_crop(redTomatoMask,[50,50],self.pickUV)

        # imgplot = plt.imshow(suctionCheckBox)
        # plt.show()
        suctionBool =  1 if (np.count_nonzero(suctionCheckBox))/2500 < 0.95 else 0
        return SelectTomatoResponse([0,suctionBool,0])
    
    def GetTomato(self):

        color_image, depth_image    = self.color_image, self.depth_image
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        

        visual_color_image = np.copy(color_image)
        # tomatoMask,tomatoC = self.getMask("red",color_image)
        # # coreMask,coreC = self.getMask("green",color_image)

        ##Alwats Detected
        detectedImage = self.tomatoDetect(color_image,depth_image,isUpdateValue=False)
        # print(self.fullRedTomato.number)
        # tomatoMask = self.fullRedTomato.masks[0] if self.fullRedTomato.number > 0 else np.ones([self.H, self.W, 1], dtype=np.uint8)
        # masked_tomato = np.zeros_like(color_image)
        # masked_tomato[np.nonzero(tomatoMask)] = color_image[np.nonzero(tomatoMask)]
        # redTomatoMask = self.getMask("red",masked_tomato)
        # masked_redTomato = np.zeros_like(color_image)
        # masked_redTomato[np.nonzero(redTomatoMask)] = color_image[np.nonzero(redTomatoMask)]
        ####
        # imgplot = plt.imshow(tomatoMask)
        # plt.show()
        # masked_color = np.zeros_like(color_image)
        # masked_depth = np.zeros_like(depth_image)
             
        # masked_color[np.nonzero(tomatoMask)] = color_image[np.nonzero(tomatoMask)]
        # masked_depth[np.nonzero(tomatoMask)] = depth_image[np.nonzero(tomatoMask)]
        # # ================================================================================ CHANGED ============
        # # converted to 16bit
        # #depth_image = depth_image.astype(np.uint16)
        # #print(self.suckUV)
        # #collisionCheckBlock = self.cell_neighbors(color_image,self.suckUV[0],self.suckUV[1],size=900)
        collisionCheckBox= self.pixel_crop(color_image,[50,50],self.pickUV)
        cv2.circle(visual_color_image,(self.pickUV[0],self.pickUV[1]), 20, (0,0,255), -1)
        # imgplot = plt.imshow(depth_image)
        # plt.show()
        #print(depth_image.dtype)
        cv2.imshow("HybridMaskCascade", detectedImage )
        #cv2.imshow("RGB", color_image)
        #cv2.imshow("mask", masked_redTomato)
        cv2.imshow("collision", collisionCheckBox)
        cv2.imshow("pickCAm", visual_color_image)
        cv2.waitKey(1)

    def process(self):
        rospy.init_node('markerFinder', anonymous=True)
        #r = rospy.Rate(60)
        rospy.Subscriber(self.CAMINFO['topic'], self.CAMINFO['msg'], self.camInfoCallback)
        rospy.Subscriber(self.COLOR['topic'], self.COLOR['msg'], self.colorCallback)
        rospy.Subscriber(self.DEPTH['topic'], self.DEPTH['msg'], self.depthCallback)
        #rospy.Subscriber(self.PC['topic'],    self.PC['msg'],    self.pcCallback)
        
        #self.depth_scale    = 0.0010000000474974513
        self.findTomatoService = rospy.Service("hand_find_tomato", SelectTomato, self.findTomatoCallBack)
        self.collisionCheckingService = rospy.Service("hand_collision_tomato", SelectTomato, self.collisionCheckingCallBack)

        ###publisher
        self.tomatoPC_pub = rospy.Publisher('/hand_tomatoPC', PointCloud2, queue_size=1)

        self.point_pub = rospy.Publisher('/hand_tomato_pick_pos', PointStamped,
                                          queue_size=1)
        
        self.marker_pub = rospy.Publisher('/hand_goal_point', Marker,
                                          queue_size=1)


        while not rospy.is_shutdown():
                if self.isCamInfo:
                    self.GetTomato()


         
if __name__ == '__main__':
    try:
        _depthImageHandler = DepthImageHandler()
        _depthImageHandler.process()

    except rospy.ROSInterruptException:
        pass
