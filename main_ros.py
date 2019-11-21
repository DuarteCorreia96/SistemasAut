#! /usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from aruco_pose_subscriber import *

from ekf import Measurement, EKF_SLAM, MotionModel

class Aruco():
    @staticmethod
    def callback(data):
        Aruco.list = []
        for i in range(len(data)):
            aruco_id = data.markers[i].id
            r,theta  = convert_pose(data.markers[i].pose.pose.position.x, data.markers[i].pose.pose.position.y)
            Aruco.list.append(Measurement(aruco_id, r, theta))


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

        Odom.dt = (new_cur_time - Odom.cur_time)/1e-9
        Odom.cur_time = new_cur_time


def main():

    ekf = EKF_SLAM(MotionModel)
    ekf.print_state()

    rospy.init_node('pose_listener', anonymous=True)
    rospy.Subscriber('/pioneer/pose', Odometry, Odom.retrieveOdom)
    rospy.init_node('aruco_pose_listener', anonymous=True)
    rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, Aruco.callback)

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():

        ekf.prediction_step(Odom.v, Odom.w, Odom.dt)
        ekf.print_state()

        #measurs = [Measurement(1, 2, 0.4), Measurement(2, 2, 0.25)]
        ekf.update_step(Aruco.list)
        ekf.print_state()

        rate.sleep()


if __name__ == "__main__":
    main()