import numpy as np
import random

from base_ekf import MotionModel, Measurement
from ekf import EKF_SLAM

class Matlab_EKF():

    def __init__(self, Q_diag, sigma, L):

        MotionModel.L = L
        Q = np.diag(Q_diag)
        MotionModel.sigma = sigma

        self.ekf = EKF_SLAM(MotionModel, Q = Q)

    def update_step(self, aruco_id, r, theta):

        measurements = [Measurement(aruco_id, r, theta)]
        self.ekf.update_step(measurements)

    def prediction_step(self, v, w, dt):

        self.ekf.prediction_step(v, w, dt)

    def get_estimate(self):

        return self.ekf.estimate

    def get_covariance(self):

        return self.ekf.covariance