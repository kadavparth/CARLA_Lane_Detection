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

cwd = os.getcwd()

raw_images_path = cwd + '/dataset/raw_images'
labels_path = cwd + '/dataset/labels'


class Listener: 

    bridge = CvBridge()
    
    def __init__(self, topic_name, data_class):
        
        self.topic_name = topic_name
        self.data_class = data_class
        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)
        
        self.buffer_size = 15 # Number of previous frames to keep track of
        self.left_fit_buffer = deque(maxlen=self.buffer_size)
        self.right_fit_buffer = deque(maxlen=self.buffer_size)
        self.frame_ct = 0
    
    def callback(self,img_msg):

        try:
            
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            warped1, Minv1, unwarped1 = win.Perspective(img)

            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            _ , sxbinary = edge.threshold(hls[:,:,1], thresh=(47,255))
            sxbinary = edge.blur_gaussian(sxbinary, kernel_size=5)
            
            warped, Minv, unwarped = win.Perspective(sxbinary)
            warped = self.contors(warped)

            hist = win.calc_hist(warped)

            left_fit, right_fit, left_fitx, right_fitx, midx, ploty, out_img, frame_sliding_window = win.get_lane_line_indices_sliding_window(warped,hist)

            blank_img = np.zeros_like(img)

            for i in range(len(ploty)):
                
                cv2.line(blank_img,(int(left_fitx[i]),int(ploty[i])), (int(left_fitx[i]),int(ploty[i])), (255,255,255),5)
                cv2.line(blank_img,(int(right_fitx[i]),int(ploty[i])), (int(right_fitx[i]),int(ploty[i])), (255,255,255),5)

            cv2.imshow('Raw Image',warped1)
            cv2.imshow('Lane Lines', blank_img)

            cv2.imwrite(f"{raw_images_path}" + "/" f"{self.frame_ct}.png", warped1)
            cv2.imwrite(f"{labels_path}" + "/" f"{self.frame_ct}.png", blank_img)

            self.frame_ct += 1

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)
        
    # functions to get hsv image, ROI, canny edge and hough transforms. 

    def contors(self,warped_image):
        
        ret,thresh = cv2.threshold(warped_image, 20,255,cv2.THRESH_BINARY)
        contors, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # warped_image = cv2.merge((warped_image,warped_image,warped_image))
        warped_cp = warped_image.copy()

        x = list(map(lambda x : len(x), contors))
        x = sorted(x)
        ls = []
        for i in contors:
            if len(i) == x[-1]:
                ls.append(i)
            elif len(i) == x[-2]:
                ls.append(i)
            else:
                continue

        cv2.drawContours(warped_cp, ls, -1, (255,255,255), cv2.FILLED)

        return warped_cp

    
if __name__ == "__main__":

    if not os.path.exists(raw_images_path) and not os.path.exists(labels_path):        
        os.makedirs(raw_images_path)
        os.makedirs(labels_path)
        print('New directories created')

    rospy.init_node('Carla_image_viewer', anonymous=True)
    topic_name = '/carla/ego_vehicle/rgb_front/image'
    data_class = Image 

    ls = Listener(topic_name, data_class)
    rospy.spin()   


