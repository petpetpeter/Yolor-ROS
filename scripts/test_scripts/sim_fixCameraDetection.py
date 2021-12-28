#!/usr/bin/env python

import numpy as np
import os
import math
import time
import cv2

import struct

from numpy.core.numeric import NaN
from cv_bridge import CvBridge
bridge = CvBridge()

import rospy
import ros_numpy
from std_msgs.msg import Header,String
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, Image
from std_msgs.msg import Bool,Float32MultiArray
from visualization_msgs.msg import Marker,MarkerArray

import glob
from geometry_msgs.msg import Point,Pose,PoseStamped,PoseArray,PointStamped
import ctypes

from realsense2_camera.srv import SelectTomato,SelectTomatoResponse
import open3d as o3d

import matplotlib.pyplot as plt



#import cv2.aruco as aruco

def nothing():
    pass

class DepthImageHandler(object):
    def __init__(self):
        self.CAMINFO = {'topic': '/rsd435/color/camera_info', 'msg': CameraInfo}
        self.isCamInfo = False

        self.COLOR = {'topic': '/rsd435/color/image_raw', 'msg': Image}
        self.DEPTH = {'topic': '/rsd435/depth/image_raw', 'msg': Image}
        self.PC = {'topic': '/tomatoPC', 'msg': PointCloud2}
        
        self.H = 720
        self.W = 1280
        self.cropSize = 1 #1 for no crop
        self.header = Header()
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
        numpyImage = ros_numpy.numpify(msg)
        numpyImage = np.nan_to_num(numpyImage, copy=True, nan=0.0)
        self.depth_image = numpyImage
    
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

    def depthToPointsDistance(self, depth_image, U, V):
        V =  np.clip(V,0,self.height-1)
        U =  np.clip(U,0,self.width-1)  
        
        x = (U - self.K[2]/self.cropSize)/self.K[0]
        y = (V - self.K[5]/self.cropSize)/self.K[4]     
        U,V = self.find_nearest_nonzero(depth_image,(U,V))
        #print(vv,uu)
        z = depth_image[V,U] / 1000.0
        x *= z
        y *= z
        # print(U,V,x,y,z)
        point3 = [z, -x, -y]
        return point3
        # Checks if a matrix is a valid rotation matrix.
    
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

        z = depth_image.flatten() 
        

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
        points = np.stack((z, -x, -y,rgb_int), axis = -1) #swap Axis Coz gazebo wrong rotation
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
        point.header.frame_id = "rgb"


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
        # imgplot = plt.imshow(hsv)
        # plt.show()
        if color == "all":
            l_b = np.array([0, 10, 0])
            u_b = np.array([50, 255, 150])
        if color == "red":
            l_b = np.array([0, 10, 0])
            u_b = np.array([50, 255, 150])
        if color == "green":
            l_b = np.array([10, 10, 0])
            u_b = np.array([255, 255, 150])

        mask_img = cv2.inRange(hsv, l_b, u_b)
        cX,cY = 0,0
        # contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours) != 0:
        #     c = max(contours, key = cv2.contourArea)
        #     M = cv2.moments(c)
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        return mask_img,(cX,cY)

    def center_crop(self,img, dim):
        width, height = img.shape[1], img.shape[0]
        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
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

        return [radius, C[0][0], C[1][0], C[2][0]]
    
    def GetTomato(self):

        color_image, depth_image    = self.color_image, self.depth_image

        tomatoMask,tomatoC = self.getMask("all",color_image)
        coreMask,coreC = self.getMask("green",color_image)

        masked_color = np.zeros_like(color_image)
        masked_depth = np.zeros_like(depth_image)
        
        masked_color[np.nonzero(tomatoMask)] = color_image[np.nonzero(tomatoMask)]
        masked_depth[np.nonzero(tomatoMask)] = depth_image[np.nonzero(tomatoMask)]
        # ================================================================================ CHANGED ============
        # converted to 16bit
        #depth_image = depth_image.astype(np.uint16)
        
        # imgplot = plt.imshow(depth_image)
        # plt.show()
        #print(depth_image.dtype)
        # cv2.imshow("image", color_image)
        # cv2.imshow("mask", masked_depth)
        # cv2.imshow("core", coreMask)
        #cv2.imshow("heta", coreMask)
        cv2.waitKey(1)



    def findTomatoCallBack(self,req):
        print(req.gripperPos)
        color_image, depth_image    = self.color_image, self.depth_image

        tomatoMask,tomatoC = self.getMask("all",color_image)
        coreMask,coreC = self.getMask("green",color_image)

        masked_color = np.zeros_like(color_image)
        masked_depth = np.zeros_like(depth_image)
        
        masked_color[np.nonzero(tomatoMask)] = color_image[np.nonzero(tomatoMask)]
        masked_depth[np.nonzero(tomatoMask)] = depth_image[np.nonzero(tomatoMask)]

        
        tomatoPoints = self.depthToPoints(masked_depth,color_image)
        tomatoChannel = np.moveaxis(tomatoPoints, 0, 1)
        tomatoX = tomatoChannel[0]
        tomatoY = tomatoChannel[1]
        tomatoZ = tomatoChannel[2]
        sphere = self.sphereFit(tomatoX,tomatoY,tomatoZ)
        #print(sphere)
        tomatoCenter = [sphere[1],sphere[2],sphere[3]]

        coreCenter = self.depthToPointsDistance(depth_image,coreC[0],coreC[1])
        self.publishPointCloud(self.tomatoPC_pub,tomatoPoints)
        self.publishPoint3(tomatoCenter)
        return SelectTomatoResponse(tomatoCenter)

    def process(self):
        rospy.init_node('markerFinder', anonymous=True)
        #r = rospy.Rate(60)
        rospy.Subscriber(self.CAMINFO['topic'], self.CAMINFO['msg'], self.camInfoCallback)
        rospy.Subscriber(self.COLOR['topic'], self.COLOR['msg'], self.colorCallback)
        rospy.Subscriber(self.DEPTH['topic'], self.DEPTH['msg'], self.depthCallback)
        #rospy.Subscriber(self.PC['topic'],    self.PC['msg'],    self.pcCallback)
        
        #self.depth_scale    = 0.0010000000474974513
        self.findTomatoService = rospy.Service("find_tomato", SelectTomato, self.findTomatoCallBack)

        ###publisher
        self.tomatoPC_pub = rospy.Publisher('/tomatoPC', PointCloud2, queue_size=1)

        self.point_pub = rospy.Publisher('/tomato_pick_pos', PointStamped,
                                          queue_size=1)
        
        self.marker_pub = rospy.Publisher('/goal_point', Marker,
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
