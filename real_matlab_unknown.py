import numpy as np
import random

from base_ekf import MotionModel, Measurement
from synth_base import Robot
from ekf_unknown import EKF_SLAM_unknown

class Matlab_EKF():

    def __init__(self, Q_diag, sigma, L, alpha):

        MotionModel.L = L
        Q = np.diag(Q_diag)
        MotionModel.sigma = sigma

        self.robot_noise = Robot()
        self.ekf = EKF_SLAM_unknown(MotionModel, Q = Q, alpha = alpha)

    def update_step(self, aruco_id, r, theta):

        measurements = [Measurement(aruco_id, r, theta)]
        self.ekf.update_step(measurements)

    def prediction_step(self, v, w, dt):

        self.robot_noise.move(v, w, dt)
        self.ekf.prediction_step(v, w, dt)

    def get_estimate(self):

        return self.ekf.estimate

    def get_covariance(self):

        return self.ekf.covariance

    def get_odom(self):

        return np.array(self.robot_noise.current, ndmin=2)