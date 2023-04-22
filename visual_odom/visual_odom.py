#!/usr/bin/env python3

import rospy
import cv2 
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
import carla 
from utils import * 


rospy.init_node('Visual_Odom')

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
carla_map = world.get_map()


class Listener: 

    bridge = CvBridge() # object for bridge
    
    def __init__(self):

        # get params for camera 
        camera_msg = rospy.wait_for_message('/carla/ego_vehicle/rgb_front/camera_info', CameraInfo)

        # intrinsic camera matrix 
        self.K = np.array(camera_msg.K).reshape(3,3)

        # subsrciber for camera 
        rospy.Subscriber('/carla/ego_vehicle/rgb_front/image', Image, self.img_callback)

        # subscriber for depth image 
        rospy.Subscriber('/carla/ego_vehicle/depth_front/image', Image, self.depth_callback)

        self.ct = 0
        self.i = 0

        self.kp_list = []
        self.des_list = []


    def img_callback(self,img_msg):

        try:
            
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            kp, des = extract_features(img)

            self.kp_list.append(kp)
            self.des_list.append(des)

            if self.ct == 0: 

                pass 

            else: 

                match = match_features(des, self.des_list[self.i])
                print(len(match))
                print(self.ct, self.i)
                self.i += 1

            self.ct += 1

            cv2.imshow('img',img)
            cv2.waitKey(0)

        except CvBridgeError as e:
            print(e)

    def depth_callback(self,msg):
        
        self.depth_msg = Listener.bridge.imgmsg_to_cv2(msg)


if __name__ == "__main__":

    ls = Listener()
    rospy.spin()   


