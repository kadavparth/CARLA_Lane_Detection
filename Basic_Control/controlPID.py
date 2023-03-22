#! /usr/bin/env python3

import rospy
import carla
import numpy as np
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaActorList
from ackermann_msgs.msg import AckermannDrive
import random

rospy.init_node("waypoints")
client = carla.Client("localhost", 2000)
client.set_timeout(2.0)
world = client.get_world()
carlaMap = world.get_map()

actors = rospy.wait_for_message("carla/actor_list", CarlaActorList)

for actor in actors.actors:
    if actor.rolename == "ego_vehicle":
        ego_vehicle = world.get_actor(actor.id)

transform = ego_vehicle.get_transform()
waypoint = carlaMap.get_waypoint(transform.location)

ptsx = []
ptsy = []

for d in np.arange(1, 100, 0.5):

    current_waypoint = waypoint.next(d)[0]

    ptsx.append(current_waypoint.transform.location.x)
    ptsy.append(current_waypoint.transform.location.y)


class Listener:
    def __init__(self, *args, **kwargs):

        self.topic_name = topic_name
        self.data_class = data_class

        self.control = carla.VehicleControl()
        self.control.throttle = 0.2
        self.control.brake = 0.0
        self.control.steer = 0.0

        self.kp, self.ki, self.kd = 0.5, 0.2, 0.07

        self.last_error = 0.0
        self.integral = 0.0

        msg = rospy.wait_for_message(topic_name, data_class)
        self.previous_time = msg.header.stamp.to_sec()

        self.sub = rospy.Subscriber(self.topic_name, self.data_class, self.callback)

    def callback(self, msg):

        transform1 = ego_vehicle.get_transform()
        statep = transform1.location
        stater = transform1.rotation

        target_idx, error_front_axle = self.calc_target_idx(statep, stater, ptsx, ptsy)

        current_time = msg.header.stamp.to_sec()
        dt = current_time - self.previous_time
        self.previous_time = current_time

        error = error_front_axle
        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        proportional = self.kp * error
        integral = self.ki * self.integral
        derivative = self.kd * derivative

        output = proportional + integral + derivative

        self.control.steer = output

        self.last_error = error

        ego_vehicle.apply_control(self.control)

        # for x, y in zip(ptsx, ptsy):
        #     world.debug.draw_point(
        #         carla.Location(x, y, 0.3),
        #         size=0.05,
        #         color=carla.Color(r=0, g=255, b=0),
        #         life_time=1.0,
        #     )

    def calc_target_idx(self, statep, stater, cx, cy):

        fx = statep.x
        fy = statep.y

        dx = [fx - icx for icx in cx]
        dy = [fy - icy for icy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        front_axle_vec = [
            -np.cos(np.deg2rad(stater.yaw) + np.pi / 2),
            -np.sin(np.deg2rad(stater.yaw) + np.pi / 2),
        ]

        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle


if __name__ == "__main__":

    topic_name = "/carla/ego_vehicle/odometry"
    data_class = Odometry

    ls = Listener(topic_name, data_class)
    rospy.spin()
