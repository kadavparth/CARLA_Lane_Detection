#! usr/bin/env/python3

import carla 
from carla_msgs.msg import CarlaActorList, CarlaActorInfo 
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
import rospy
import tf 
from tf.transformations import euler_from_quaternion
import numpy as np 
import message_filters
import cv2 

rospy.init_node('Carla_image_viewer', anonymous=True)

client = carla.Client("localhost", 2000)
client.set_timeout(2.0)
world = client.get_world()

actors = rospy.wait_for_message('carla/actor_list', CarlaActorList)

for actor in actors.actors:

    if actor.rolename == 'ego_vehicle':
        ego_veh = world.get_actor(actor.id)
    elif actor.rolename == 'rgb_front':
        camera_mat = world.get_actor(actor.id)
    elif actor.rolename == 'hero':
        lead_veh = world.get_actor(actor.id)


class Listener:

    camera_params = rospy.wait_for_message('/carla/ego_vehicle/rgb_front/camera_info', CameraInfo)
    bridge = CvBridge()

    def __init__(self, topic_name, data_class):
        
        self.topic_name = topic_name
        self.data_class = data_class
        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)

    def callback(self,img_msg):

        try:
            img = Listener.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            lead_veh_transform = lead_veh.get_transform()
            camera_mat_transform = np.array([camera_mat.get_transform().get_matrix()])

            lead_veh_position = np.array([lead_veh_transform.location.x,  lead_veh_transform.location.y, lead_veh_transform.location.z, 1]).reshape(4,1)

            local_hero = np.linalg.inv(camera_mat_transform) @ lead_veh_position

            opencv_local_hero = np.array([[0,1,0],[0,0,-1],[1,0,0]]).reshape(3,3) @ local_hero[0][:3]

            pixel_coords = np.array([Listener.camera_params.K]).reshape(3,3) @ opencv_local_hero

            pixel_coords[0] = pixel_coords[0]/pixel_coords[2]
            pixel_coords[1] = pixel_coords[1]/pixel_coords[2]
            
            pixel_coords = [pixel_coords[0], pixel_coords[1]]

            point_img = cv2.circle(img, (int(pixel_coords[0]), int(pixel_coords[1])), 3, (0,0,255), -1)

            cv2.imshow('frame',point_img)

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

    
if __name__ == "__main__":

    
    topic_name = '/carla/ego_vehicle/rgb_front/image'
    data_class = Image 

    ls = Listener(topic_name, data_class)
    rospy.spin()   



