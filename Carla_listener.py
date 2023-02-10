#! usr/bin/env/python3

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

class Listener: 

    bridge = CvBridge()

    def __init__(self, topic_name, data_class):
        
        self.topic_name = topic_name
        self.data_class = data_class
        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)
        
        self.buffer_size = 15 # Number of previous frames to keep track of
        self.left_fit_buffer = deque(maxlen=self.buffer_size)
        self.right_fit_buffer = deque(maxlen=self.buffer_size)
    
    def callback(self,img_msg):

        try:
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            _ , sxbinary = edge.threshold(hls[:,:,1], thresh=(47,255))
            sxbinary = edge.blur_gaussian(sxbinary, kernel_size=5)
            # masked_roi_img = self.roi_mask(sxbinary)
            
            warped, Minv, unwarped = win.Perspective(sxbinary)
            warped = self.contors(warped)

            hist = win.calc_hist(warped)

            left_fit, right_fit, left_fitx, right_fitx, midx, ploty, out_img, frame_sliding_window = win.get_lane_line_indices_sliding_window(warped,hist)


            ## For first method 

            # self.left_fit_buffer.append(left_fit)
            # self.right_fit_buffer.append(right_fit)

            # left_fit_avg = np.mean(self.left_fit_buffer, axis=0)
            # right_fit_avg = np.mean(self.right_fit_buffer, axis=0)

            blank_img = np.zeros_like(img)
            
            ## For first method 

            # left_fitx = left_fit_avg[0] * ploty**2 + left_fit_avg[1] * ploty + left_fit_avg[2] 
            # right_fitx = right_fit_avg[0] * ploty**2 + right_fit_avg[1] * ploty + right_fit_avg[2]
            # middle_fitx = middle_fitx[0] * plotly**2 + middle_fitx[1] * plotly + middle_fitx[2]

            for i in range(len(ploty)):
                
                cv2.line(blank_img,(int(left_fitx[i]),int(ploty[i])), (int(left_fitx[i]),int(ploty[i])), (0,255,0),3)
                cv2.line(blank_img,(int(right_fitx[i]),int(ploty[i])), (int(right_fitx[i]),int(ploty[i])), (0,0,255),3)
                cv2.line(blank_img,(int(midx[i]), int(ploty[i])), (int(midx[i]), int(ploty[i])), (255,0,0),2)

            newwarp = cv2.warpPerspective(out_img, Minv, (800,800))
            img = cv2.resize(img, (500,500))
            newwarp = cv2.resize(newwarp, (500,500))

            newwarp1 = cv2.addWeighted(newwarp, 0.5, img, 0.7,0)
            newwarp1 = cv2.resize(newwarp1, (500,500))
            
            lane_proj = np.concatenate([img,newwarp1],axis=1)

            out_img = cv2.resize(out_img, (500,500))
            blank_img = cv2.resize(blank_img, (500,500))

            out_img1 = np.concatenate([out_img, blank_img],axis=1)

            final_img = np.concatenate([lane_proj,out_img1],axis=0)

            cv2.imshow('Warped Sliding Window + Polyfit Lane Lines', final_img)
            cv2.imshow('frame',frame_sliding_window)
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

    def hsv_image(self,image):

        lowerwhite = (0,0,100)
        upperwhite = (172,111,255)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerwhite, upperwhite)

        return mask 
    
    def roi_mask(self,image):
        self.mask_vertices = np.array([[0,800], [250,475], [480,475], [800,800]])
        mask_vertices = self.mask_vertices
        blank_img = np.zeros_like(image)
        blank_img = cv2.fillPoly(blank_img, [mask_vertices], 255)
        masked_roi_image = cv2.bitwise_and(image,blank_img)

        return masked_roi_image
    
    def canny_hough(self,image):
        dst = cv2.Canny(image, 50, 200, None, 3)
        cdst = dst.copy()
        cdst = image.copy()
        lines = cv2.HoughLinesP(cdst, 1, np.pi/60, 100, np.array([]), 5, 50)
        if lines is not None:
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(cdst, (x1,y1), (x2,y2), (255,0,255), thickness=3)
        return cdst 
    
if __name__ == "__main__":

    rospy.init_node('Carla_image_viewer', anonymous=True)
    topic_name = '/carla/ego_vehicle/rgb_front/image'
    data_class = Image 

    ls = Listener(topic_name, data_class)
    rospy.spin()   


