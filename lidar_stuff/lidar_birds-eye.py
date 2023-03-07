#! usr/bin/env/python3

import rospy
import cv2 
from sensor_msgs.msg import Image, PointCloud2
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imshow

rospy.init_node('Carla_lidar_viewer', anonymous=True)

class Listener: 

    def __init__(self, topic_name, data_class):
        
        self.topic_name = topic_name
        self.data_class = data_class
        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)

        self.grid_res =  0.2
        self.grid_height = 40
        self.grid_width = 40 

        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0


    def callback(self,msg):

        point_cloud = np.frombuffer(msg.data, dtype=np.float32)
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))

        # np.save('point_cloud.npy', point_cloud)

        self.min_x = np.min(point_cloud[:, 0])
        self.max_x = np.max(point_cloud[:, 0])
        self.min_y = np.min(point_cloud[:, 1])
        self.max_y = np.max(point_cloud[:, 1])

        grid_width = int(np.ceil((self.max_x - self.min_x) / self.grid_res))
        grid_height = int(np.ceil((self.max_y - self.min_y) / self.grid_res))

        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        grid_x = ((point_cloud[:, 0] - self.min_x) / self.grid_res).astype(np.int32)
        grid_y = ((point_cloud[:, 1] - self.min_y) / self.grid_res).astype(np.int32)

        grid[grid_y, grid_x] = 1

        grid = np.expand_dims(grid, axis=2)

        cv2.imshow('grid',255*grid)
        cv2.waitKey(1)


if __name__ == "__main__":

    topic_name = '/carla/ego_vehicle/lidar'
    data_class = PointCloud2 

    ls = Listener(topic_name, data_class)
    rospy.spin()   
    cv2.destroyAllWindows()