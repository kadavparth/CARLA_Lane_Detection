#! usr/bin/env/python3

import rospy
import cv2 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 


class Listener: 

    bridge = CvBridge()

    def __init__(self, topic_name, data_class):
        
        self.topic_name = topic_name
        self.data_class = data_class
        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)
    
    def callback(self,img_msg):

        # RGB (157,234,50)

        try:
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "rgb8")

            bg = np.where( (img[:,:,0] != 157) &
                           (img[:,:,1] != 234) &
                           (img[:,:,2] != 50) )
            
            img[bg] = [0,0,0]

            lane_line_px = np.where( (img[:,:,0] != 0) &
                           (img[:,:,1] != 0) &
                           (img[:,:,2] != 0))
            
            img[lane_line_px] = [255,255,255]
            
            mask = self.roi_mask(img)

            warped, Minv, unwarped = self.Perspective(mask) 

            lane_lines = self.find_lane_lines(warped)

            warp_new = cv2.warpPerspective(lane_lines, Minv, (800,800))

            cv2.imshow('ll',img)
            cv2.imshow('ll1',warp_new)
            
            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

    def roi_mask(self,image):

        self.mask_vertices = np.array([[0,800], [250,475], [480,475], [800,800]])
        mask_vertices = self.mask_vertices
        blank_img = np.zeros_like(image)
        blank_img = cv2.fillPoly(blank_img, [mask_vertices], 255)
        masked_roi_image = cv2.bitwise_and(image,blank_img)

        return masked_roi_image
    
    def Perspective(self,img):
    
        # src = np.float32([[0,745], [150,481], [580,481], [800,745]]) # works well
        src = np.float32([[0,800], [90,481], [630,481], [800,800]])
        dst = np.float32([[0,img.shape[0]], [0,0], [img.shape[1], 0], [img.shape[1], img.shape[0]]])
        
        M = cv2.getPerspectiveTransform(src,dst)
        
        Minv = cv2.getPerspectiveTransform(dst,src)
        
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        unwarped = cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        return warped, Minv, unwarped

    def find_lane_lines(self,warped):
        left_half = warped[:,:400,:]
        right_half = warped[:,400:,:]

        left_half_nonzero = left_half.nonzero()
        right_half_nonzero = right_half.nonzero()

        left_halfx = np.array(left_half_nonzero[1])
        left_halfy = np.array(left_half_nonzero[0])

        right_halfx = np.array(right_half_nonzero[1])
        right_halfy = np.array(right_half_nonzero[0])

        left_fit = np.polyfit(left_halfy, left_halfx, 2)
        right_fit = np.polyfit(right_halfy, right_halfx, 2) 

        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


        blank_img_l = np.zeros_like(left_half)
        blank_img_r = np.zeros_like(right_half)

        for i in range(len(ploty)):
            
            cv2.line(blank_img_l,(int(left_fitx[i]),int(ploty[i])), (int(left_fitx[i]),int(ploty[i])), (255,255,255),5)
            cv2.line(blank_img_r,(int(right_fitx[i]),int(ploty[i])), (int(right_fitx[i]),int(ploty[i])), (255,255,255),5)

        return np.concatenate([blank_img_l,blank_img_r], axis=1)

if __name__ == "__main__":

    rospy.init_node('Carla_image_viewer', anonymous=True)
    topic_name = '/carla/ego_vehicle/semantic_segmentation_front/image'
    data_class = Image 

    ls = Listener(topic_name, data_class)
    rospy.spin() 