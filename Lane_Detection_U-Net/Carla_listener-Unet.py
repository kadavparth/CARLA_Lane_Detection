#!/usr/bin/env python3

import rospy
import cv2 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
import math 
import edge_detect as edge 
import sliding_win as win 
import matplotlib.pyplot as plt
from collections import deque
import os 
from tensorflow import keras 

cwd = os.getcwd()

raw_images_path = cwd + '/dataset/raw_images'
labels_path = cwd + '/dataset/labels'

model = keras.models.load_model('/home/parth/Downloads/model.h5', compile=False)

class Listener: 

    bridge = CvBridge()
    
    def __init__(self, topic_name, data_class):
        
        self.topic_name = topic_name
        self.data_class = data_class
        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)

    
    def callback(self,img_msg):

        try:
            
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            warped1, Minv1, unwarped1 = win.Perspective(img)

            warped1 = cv2.resize(warped1, (512,512))

            warped_cp = warped1.copy()

            warped1 = np.expand_dims(warped1, axis=0)

            pred = model.predict(warped1)

            pred = (pred > 0.2).astype(np.uint8)
            pred = np.squeeze(pred*255.)

            hist = win.calc_hist(pred)

            left_fit, right_fit, left_fitx, right_fitx, midx, ploty, out_img, frame_sliding_window = win.get_lane_line_indices_sliding_window(pred,hist)

            blank_img = np.zeros_like(img)

            for i in range(len(ploty)):
                
                cv2.line(blank_img,(int(left_fitx[i]),int(ploty[i])), (int(left_fitx[i]),int(ploty[i])), (0,0,255),5)
                cv2.line(blank_img,(int(right_fitx[i]),int(ploty[i])), (int(right_fitx[i]),int(ploty[i])), (0,255,0),5)
                cv2.line(blank_img,(int(midx[i]), int(ploty[i])), (int(midx[i]), int(ploty[i])), (255,0,0),2)
            
            cv2.imshow('Raw_Img', warped_cp)
            cv2.imshow('Warped Sliding Window + Polyfit Lane Lines', blank_img[0:500, 0:500])

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)
        

if __name__ == "__main__":

    rospy.init_node('Carla_image_viewer', anonymous=True)
    topic_name = '/carla/ego_vehicle/rgb_front/image'
    data_class = Image 

    ls = Listener(topic_name, data_class)
    rospy.spin()   


