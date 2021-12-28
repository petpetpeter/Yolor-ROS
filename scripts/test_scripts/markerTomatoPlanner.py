#!/usr/bin/env python3

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
from tomato_detection.srv import SelectTomato,SelectTomatoResponse
from geometry_msgs.msg import Point,Pose,PoseStamped,PoseArray,PointStamped


class TomatoPlanner(object):
    def __init__(self):
        self.TomatoMarkerArrayTopic = {'topic': 'tomatoMarkers', 'msg': MarkerArray}
        
        self.fullRedTomato = self.Tomato(stage = "b_fully_ripened")
        self.halfRedTomato = self.Tomato(stage = "b_half_ripened")
        self.greenTomato = self.Tomato(stage = "b_green")
        self.getLabelInstanceDict = {0:self.fullRedTomato ,
                                    1:self.halfRedTomato,
                                    2:self.greenTomato}

    class Tomato():
        def __init__(self,stage):
            self.stage = stage
            self.camPos = []
        def __str__(self):
            return f"tomato class : {self.stage}\nnumber :{len(self.camPos)}"
        def getClosetTomatoPos(self,targetPos):
            if len(self.camPos) == 0:
                return [0,0,0]
            Distances = [math.dist(tomatoPos,targetPos) for tomatoPos in self.camPos]
            min_index = Distances.index(min(Distances)) 
            return self.camPos[min_index] #return min distance position from input target in camera coordinate
        def reset(self):
            self.camPos = []
    
    def tomatoArrayCallback(self, msg):
        tomatoPos3Array = [[marker.pose.position.x,marker.pose.position.y,marker.pose.position.z] for marker in msg.markers]
        self.fullRedTomato.camPos = tomatoPos3Array
    
    def findTomatoCallBack(self,req):
        print(req.gripperPos,"gripperPos")
        gripperPos_cam = req.gripperPos
        tomatoPos_cam = self.fullRedTomato.getClosetTomatoPos(gripperPos_cam)
        print(tomatoPos_cam,"tomatoPos_cam")
        return SelectTomatoResponse(tomatoPos_cam)


    def updateTomato(self):
        #print(f"Number Of Tomato is: {len(self.fullRedTomato.camPos)}")
        return 1

    def process(self):
        rospy.init_node('tomatoPlanner', anonymous=True)
        #r = rospy.Rate(60)
        rospy.Subscriber(self.TomatoMarkerArrayTopic['topic'], self.TomatoMarkerArrayTopic['msg'], self.tomatoArrayCallback)        
        self.array_pub = rospy.Publisher('/tomatoMarkers', MarkerArray,queue_size=1)

        self.findTomatoService = rospy.Service("closet_tomato", SelectTomato, self.findTomatoCallBack)
        print("Find Tomato Service is Ready...")
        while not rospy.is_shutdown():
            self.updateTomato()


         
if __name__ == '__main__':
    try:
        _planner = TomatoPlanner()
        _planner.process()

    except rospy.ROSInterruptException:
        pass
