#!/usr/bin/env python3

import rospy
import carla
import time
import numpy as np
from carla_msgs.msg import CarlaActorList
from nav_msgs.msg import Odometry
import math 

rospy.init_node('waypoints')
client = carla.Client('localhost', 2000)  
client.set_timeout(2.0)
world = client.get_world()    
carlaMap = world.get_map()

actors = rospy.wait_for_message('carla/actor_list', CarlaActorList)

for actor in actors.actors:
    if actor.rolename == 'ego_vehicle':
        ego_vehicle = world.get_actor( actor.id )

# Kp = 0.1
# Ki = 0.1
# Kd = 0.1 

# error = 0 
# int_error = 0

# waypoint_interval = 1.0

class Listener: 

    def __init__(self, *args, **kwargs):

        self.topic_name = topic_name
        self.data_class = data_class
        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)
        self.lane_width = 0

    def callback(self, msg):

        transform = ego_vehicle.get_transform()

        waypoints = carlaMap.get_waypoint(transform.location)

        points = []

        ec_distance = []

        for d in np.arange(1,10,0.1):

            current_waypoint = waypoints.next(d)[0]

            self.lane_width = waypoints.lane_width 

            # points.append(current_waypoint)

            min_distance = math.inf

            distance = math.sqrt( (current_waypoint.transform.location.x - transform.location.x)**2 + (current_waypoint.transform.location.y  - transform.location.y)**2 )

            ec_distance.append(distance)

        cte = self.lane_width / 2.0 - min(ec_distance)

        print(cte)

        # print(min(ec_distance))


if __name__ == "__main__":

    topic_name = '/carla/ego_vehicle/odometry'
    data_class =  Odometry

    ls = Listener(topic_name, data_class)
    rospy.spin()   


# while not rospy.is_shutdown(): 

#     transform = ego_veh.get_transform()

#     print(transform)

#     waypoints = carlaMap.get_waypoint(transform.location)

#     points = []

#     waypoint_distance = 1.0

#     for i in range(20):
#     # Calculate the location of the next waypoint
#         next_location = transform.location + transform.get_forward_vector() * (i+1) * waypoint_distance
#         # Get the waypoint object
#         next_waypoint = carlaMap.get_waypoint(next_location)
#     # Add the waypoint to the list
#         points.append(next_waypoint.transform.location.y)

#         # print(current_waypoint.transform.location.x, current_waypoint.transform.location.y )

#     print(points)
#     time.sleep(3)
#     # der_error = cte - error 

#     # int_error += cte 

#     # steering_angle = -Kp * cte - Kd * der_error - Ki * int_error

#     # control.throttle = 0.2 
    # control.steer = steering_angle

    # ego_veh.apply_control(control)

    # error = cte 

    # waypoint = waypoints.next(waypoint_interval)[0]

    # if waypoint is None:
    #     break 

    # waypoints = waypoint




    