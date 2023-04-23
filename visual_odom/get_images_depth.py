#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import carla

rospy.init_node("Get_data")

client = carla.Client("localhost", 2000)
client.set_timeout(2.0)
world = client.get_world()
carla_map = world.get_map()


class Listener:
    bridge = CvBridge()  # object for bridge

    def __init__(self):
        # get params for camera
        camera_msg = rospy.wait_for_message(
            "/carla/ego_vehicle/rgb_front/camera_info", CameraInfo
        )

        # intrinsic camera matrix
        self.K = np.array(camera_msg.K).reshape(3, 3)

        # subsrciber for camera
        rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, self.img_callback)

        # subscriber for depth image
        rospy.Subscriber(
            "/carla/ego_vehicle/depth_front/image", Image, self.depth_callback
        )

        self.ct = 0

        np.save("camera_mat.npy",self.K)

    def img_callback(self, img_msg):
        try:
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            cv2.imwrite(f"images/{self.ct}.jpg", img)

            np.save(f"depth_maps/{self.ct}.npy", self.depth_msg)

        except CvBridgeError as e:
            print(e)

        self.ct += 1

    def depth_callback(self, msg):

        self.depth_msg = Listener.bridge.imgmsg_to_cv2(msg)

        self.depth_msg = np.array(self.depth_msg, dtype=np.float64)


if __name__ == "__main__":
    ls = Listener()
    rospy.spin()
