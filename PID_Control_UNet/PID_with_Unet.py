#!/usr/bin/env python3

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
import os 
from tensorflow import keras 
import carla 

cwd = os.getcwd()

rospy.init_node("PID With Unet")
client = carla.Client("localhost", 2000)
client.set_timeout(2.0)
world = client.get_world()
carlaMap = world.get_map()

actors = rospy.wait_for_message("carla/actor_list", CarlaActorList)

for actor in actors.actors:
    if actor.rolename == "ego_vehicle":
        ego_vehicle = world.get_actor(actor.id)


model = keras.models.load_model('/home/eeavlab/Desktop/carla_stuff/model_512x512v6.h5', compile=False)

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
        self.control.throttle = 0.2
        self.control.brake = 0.0
        self.control.steer = 0.0

    def callback(self,img_msg):

        try:
            
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            warped1, Minv1, unwarped1 = win.Perspective(img)

            warped1 = np.expand_dims(warped1, axis=0)

            pred = model.predict(warped1)

            pred = (pred > 0.4).astype(np.float32)

            pred = np.expand_dims( (np.squeeze(pred*255.)), axis=2)

            hist = win.calc_hist(pred)

            left_fit, right_fit, left_fitx, right_fitx, midx, ploty, out_img, frame_sliding_window = win.get_lane_line_indices_sliding_window(pred,hist)

            blank_img = np.zeros((512,512,3))

            center_line_x = [img.shape[0] // 2, img.shape[1] // 2]
            center_line_y = [0, img.shape[0] * 0.25]

            center_line_fit = np.polyfit(center_line_y, center_line_x, 1)

            mid = center_line_fit[0] * ploty[0:127] + center_line_fit[1]
            
            lateral_offset = mid[0] - midx[127]

            hlaf_lane_width = 221.44

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
                cv2.line(blank_img, (256, 128), ((blank_img.shape[0] // 2) , 0), (255,255,255), 3)
                cv2.line(blank_img, (0,128), (blank_img.shape[0], 128), (255,255,255), 3)

                cv2.circle(blank_img, (int(midx[127]), int(ploty[127])), 3, (255,255,0), 5)
                cv2.circle(blank_img, (int(mid[0]), int(ploty[127])), 3, (255,255,0), 5)
                cv2.putText(blank_img, f"PID Output for steer {lateral_offset}", (50,750), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                

            cv2.imshow('Raw Image', img)
            cv2.imshow('Prediction',pred)
            cv2.imshow('Polytfitted Lane Lines', blank_img)

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)
    

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

    topic_name = '/carla/ego_vehicle/rgb_front/image'
    data_class = Image 

    ls = Listener(topic_name, data_class)
    rospy.spin()   


