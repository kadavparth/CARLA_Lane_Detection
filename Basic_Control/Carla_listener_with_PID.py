#! usr/bin/env/python3

import rospy
import cv2 
from sensor_msgs.msg import Image 
from carla_msgs.msg import CarlaActorList
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
import math 
import edge_detect as edge 
import sliding_win as win 
import matplotlib.pyplot as plt
from collections import deque
import carla 

rospy.init_node("waypoints")
client = carla.Client("localhost", 2000)
client.set_timeout(2.0)
world = client.get_world()
carlaMap = world.get_map()

actors = rospy.wait_for_message("carla/actor_list", CarlaActorList)

for actor in actors.actors:
    if actor.rolename == "ego_vehicle":
        ego_vehicle = world.get_actor(actor.id)

class Listener: 

    bridge = CvBridge()

    def __init__(self, topic_name, data_class):
        
        self.topic_name = topic_name
        self.data_class = data_class
        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)
        
        self.kp, self.ki, self.kd = 0.5, 0.2, 0.07 

        self.last_error = 0.0 
        self.integral = 0.0 

        msg = rospy.wait_for_message(topic_name, data_class)

        self.previous_time = msg.header.stamp.to_sec()

        self.control = carla.VehicleControl()
        self.control.throttle = 0.35
        self.control.brake = 0.0
        self.control.steer = 0.0
        
    
    def callback(self,img_msg):

        try:
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            _ , sxbinary = edge.threshold(hls[:,:,1], thresh=(47,255))
            sxbinary = edge.blur_gaussian(sxbinary, kernel_size=5)
            # masked_roi_img = self.roi_mask(sxbinary)
            
            warped, Minv, unwarped = win.Perspective(sxbinary)

            # warped = self.contors(warped)

            hist = win.calc_hist(warped)

            left_fit, right_fit, left_fitx, right_fitx, midx, ploty, out_img, frame_sliding_window = win.get_lane_line_indices_sliding_window(warped,hist)

            blank_img = np.zeros_like(img)

            center_line_x = [400,400]
            center_line_y = [0,200]

            center_line_fit = np.polyfit(center_line_y, center_line_x, 1)

            mid = center_line_fit[0] * ploty[0:199] + center_line_fit[1]
            
            lateral_offset = mid[0] - midx[199]

            hlaf_lane_width = 346

            lateral_offset = lateral_offset / hlaf_lane_width

            current_time = img_msg.header.stamp.to_sec()
            dt = current_time - self.previous_time
            self.previous_time = current_time

            error = lateral_offset
            self.integral += error * dt 
            derivative = (error - self.last_error) / dt 

            proportional = self.kp * error 
            integral = self.ki * self.integral
            derivative = self.kd * derivative

            output = proportional + integral + derivative

            self.control.steer = -output

            self.last_error = error

            ego_vehicle.apply_control(self.control)

            print(f"lateral offset : {lateral_offset}, PID output : {output}")
            
            for i in range(len(ploty)):
                
                cv2.line(blank_img,(int(left_fitx[i]),int(ploty[i])), (int(left_fitx[i]),int(ploty[i])), (0,255,0),3)
                cv2.line(blank_img,(int(right_fitx[i]),int(ploty[i])), (int(right_fitx[i]),int(ploty[i])), (0,0,255),3)
                cv2.line(blank_img,(int(midx[i]), int(ploty[i])), (int(midx[i]), int(ploty[i])), (255,0,0),2)
                cv2.line(blank_img, (400 , 200), ((blank_img.shape[0] // 2) , 0), (255,255,255), 3)
                cv2.line(blank_img, (0,200), (blank_img.shape[0], 200), (255,255,255), 3)

                cv2.circle(blank_img, (int(midx[199]), int(ploty[199])), 3, (255,255,0), 5)
                cv2.circle(blank_img, (int(mid[0]), int(ploty[199])), 3, (255,255,0), 5)
                

            cv2.imshow('frame',blank_img)
            # cv2.imshow('frame1', )
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

    topic_name = '/carla/ego_vehicle/rgb_front/image'
    data_class = Image 

    ls = Listener(topic_name, data_class)
    rospy.spin()   


