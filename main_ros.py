#! /usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from aruco_msgs.msg import MarkerArray

from ekf import Measurement, EKF_SLAM, MotionModel

class Aruco():
    ar_list = []
    list_to_file = []

    @staticmethod
    def callback(data):
        Aruco.ar_list = []

        time = data.header.stamp.secs + data.header.stamp.nsecs * 1e-9 
        for marker in data.markers:
            aruco_id = marker.id
            r,theta  = convert_pose(marker.pose.pose.position.x, marker.pose.pose.position.z)
            Aruco.ar_list.append(Measurement(aruco_id, r, theta))


            Aruco.list_to_file.append((time, aruco_id, r, theta))


def convert_pose(pose_x, pose_y):
    r = np.sqrt(pose_x**2 + pose_y**2)
    angle = np.arctan2(pose_y, pose_x)
    return [r, angle]

class Odom():

    cur_time = 0
    v = 0
    w = 0
    dt = 0

    @staticmethod
    def retrieveOdom(msg):
        Odom.v = msg.twist.twist.linear.x
        Odom.w = msg.twist.twist.angular.z
        new_cur_time = msg.header.stamp.nsecs

        Odom.dt = (new_cur_time - Odom.cur_time) * 1e-9
        Odom.cur_time = new_cur_time


def main():

    ekf = EKF_SLAM(MotionModel)
    ekf.print_state()

    rospy.init_node('pose_listener', anonymous=True)
    rospy.Subscriber('/pioneer/pose', Odometry, Odom.retrieveOdom)
   # rospy.init_node('aruco_pose_listener', anonymous=True)
    rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, Aruco.callback)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        # without deadzone
        ekf.prediction_step(Odom.v, Odom.w, Odom.dt)
        ekf.print_state()
        #measurs = [Measurement(1, 2, 0.4), Measurement(2, 2, 0.25)]
        ekf.update_step(Aruco.ar_list)
        ekf.print_state()

        # with deadzone 
        '''
        if  np.linalg.norm(Odom.v) > 0.1 or np.linalg.norm(Odom.w) > 0.1: 
            ekf.prediction_step(Odom.v, Odom.w, Odom.dt)
            ekf.print_state()
            ekf.update_step(Aruco.ar_list)
            ekf.print_state()
        '''
        rate.sleep()

    with open("data.txt", "w+") as file:
        for item in Aruco.list_to_file:
            file.write(str(item))

if __name__ == "__main__":
    main()