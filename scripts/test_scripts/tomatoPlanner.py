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
from geometry_msgs.msg import Point,Pose,PoseStamped,PoseArray,PointStamped,Twist
from tf.transformations import quaternion_from_euler,euler_from_quaternion


class TomatoPlanner(object):
    def __init__(self):
        self.TomatoPoseArrayTopic = {'topic': 'tomatoArray', 'msg': PoseArray}
        
        self.fullRedTomato = self.Tomato(stage = "b_fully_ripened")
        self.halfRedTomato = self.Tomato(stage = "b_half_ripened")
        self.greenTomato = self.Tomato(stage = "b_green")
        self.getLabelInstanceDict = {0:self.fullRedTomato ,
                                    1:self.halfRedTomato,
                                    2:self.greenTomato}
        self.sumrotation2rotateDict = {0: -1,
                                    -1:0,
                                    5: 1}

        self.robotStateTopic = {'topic': 'robot_state', 'msg': String}
        self.robotState = "waiting" # waiting, moving, serching, picking, stop
        
    
    class Tomato():
        def __init__(self,stage):
            self.stage = stage
            self.camPoseArray = [] #[x,y,z,zRot]
            self.num = 0
        def __str__(self):
            return f"tomato class : {self.stage}\nnumber :{len(self.camPoseArray)}"
        def get_closet_tomato_pos(self,targetPos):
            if len(self.camPoseArray) == 0:
                return [0]*7
            Distances = [math.dist(tomatoPos[:3],targetPos) for tomatoPos in self.camPoseArray]
            min_index = Distances.index(min(Distances)) 
            return self.camPoseArray[min_index] #return min distance position from input target in camera coordinate
        def reset(self):
            self.camPoseArray = []
            self.num = 0
    
    def tomato_array_callback(self, msg):
        tomatoPoseArray = []
        for pose in msg.poses:
            tomatoPos = [pose.position.x,pose.position.y,pose.position.z]
            tomatoRot = list(euler_from_quaternion([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]))
            tomatoPose = tomatoPos+ self.heta_action_from_rotation(tomatoRot)
            tomatoPoseArray.append(tomatoPose)
        self.fullRedTomato.camPoseArray = tomatoPoseArray
        self.fullRedTomato.num = len(tomatoPoseArray)
    
    def robot_state_callback(self,msg):
        self.robotState = msg.data

    def heta_action_from_rotation(self,rotation3):
        sumRotation = int(sum(rotation3))
        #print("sumRotation",sumRotation)
        robotRotation = self.sumrotation2rotateDict[sumRotation]
        return [robotRotation]

    
    def find_tomato_service_callback(self,req):
        print(req.gripperPos,"gripperPos")
        gripperPos_cam = req.gripperPos
        tomatoPos_cam = self.fullRedTomato.get_closet_tomato_pos(gripperPos_cam)
        print(tomatoPos_cam,"tomatoPos_cam")
        return SelectTomatoResponse(tomatoPos_cam)
    


    ### For Robot ARM ####
    def pick_tomatoes(self):
        self.statePublisher.publish("picking")
        t0 = time.time()
        isNotTimeOut = False
        while time.time() - t0 < 20 and self.robotState == "picking":
            print(f"Picking Tomato for:{time.time()-t0:.2f} second")
            if self.fullRedTomato.num == 0:
                print("No more Tomato")
                isNotTimeOut = True
                break
        if not isNotTimeOut:
            print("Time Out")
        return 1
    
    ### For Dolly ####
    def get_vel(self,vel3List):
        move_cmd = Twist()
        move_cmd.linear.x = vel3List[0]
        move_cmd.linear.y = vel3List[1]
        move_cmd.angular.z = vel3List[2]
        return move_cmd

    def go_to_start_point(self):
        self.statePublisher.publish("moving")
        #publish "start" to dolly
        #wait for "finish" message from dolly
        finish = rospy.wait_for_message
        return 1

    def find_tomatoes(self,sign = 1):
        self.statePublisher.publish("searching")
        maxVel = 0.2
        maxDist = 0.3
        moveMsg = self.get_vel([maxVel*sign,0,0]) # x y theta
        self.movingVelPublisher.publish(moveMsg)
        
        t0 = time.time()
        isNotTimeOut = False
        try:
            while time.time() - t0 < 10 and self.robotState == "searching":
                print(f"Searching Tomato for:{time.time()-t0:.2f} second")
                if self.fullRedTomato.num != 0 and abs(self.fullRedTomato.camPoseArray[0][0]) > 0.001:
                    tomatoX = self.fullRedTomato.camPoseArray[0][0]
                    clippedX = max(-maxDist, min(maxDist, tomatoX))
                    updateVel = -1 * sign * np.interp(clippedX,[-maxDist,maxDist],[-maxVel,maxVel])
                    print(f"Found Tomato at {clippedX:.2f} m")
                    print(f"Going Tomato at {updateVel:.4f} m/s")
                    moveMsg = self.get_vel([updateVel,0,0]) # x y theta
                    self.movingVelPublisher.publish(moveMsg)
                    if abs(clippedX) < 0.05:
                        moveMsg = self.get_vel([0,0,0]) # x y theta
                        self.movingVelPublisher.publish(moveMsg)
                        isNotTimeOut = True
                        break
        except KeyboardInterrupt:
            pass
        if not isNotTimeOut:
            print("Time Out")
        moveMsg = self.get_vel([0,0,0]) # x y theta
        self.movingVelPublisher.publish(moveMsg)

    def process(self):
        rospy.init_node('tomatoPlanner', anonymous=True)
        #r = rospy.Rate(60)
        rospy.Subscriber(self.TomatoPoseArrayTopic['topic'], self.TomatoPoseArrayTopic['msg'], self.tomato_array_callback) 
        rospy.Subscriber(self.robotStateTopic['topic'], self.robotStateTopic['msg'], self.robot_state_callback)        
        self.statePublisher = rospy.Publisher('robot_state', String,queue_size=1)
        self.movingVelPublisher = rospy.Publisher('base_velocity_controller/cmd_vel', Twist,queue_size=1)

        self.findTomatoService = rospy.Service("closet_tomato", SelectTomato, self.find_tomato_service_callback)
        print("Find Tomato Service is Ready...")
        input("Press Any Key to Continue...")
        
        #self.go_to_start_point()
        while not rospy.is_shutdown() and self.robotState != "stop":
            self.find_tomatoes()
            #self.pick_tomatoes()
            time.sleep(1)



         
if __name__ == '__main__':
    try:
        _planner = TomatoPlanner()
        _planner.process()

    except rospy.ROSInterruptException:
        pass
