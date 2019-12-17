import numpy as np
import random

from ekf_unknown import EKF_SLAM_unknown, MotionModel
from synth_base import Landmark, Robot

class Matlab_EKF():

    def __init__(self, Q_diag, sigma, mu_odom, mu_observation, alpha = 50):

        Q = np.diag(Q_diag)
        MotionModel.sigma = sigma

        # Measurement noises
        self.mu_v = mu_odom[0]
        self.mu_w = mu_odom[1]

        Landmark.mu_r     = mu_observation[0]
        Landmark.mu_theta = mu_observation[1]

        self.robot = Robot()
        self.robot_noise = Robot()
        self.ekf = EKF_SLAM_unknown(MotionModel, Q = Q, alpha = alpha)

        self.landmarks = []
        self.landmarks_x = []
        self.landmarks_y = []

    def add_landmark(self, mark_id, x, y):

        self.landmarks.append(Landmark(mark_id, x, y))
        self.landmarks_x.append(x)
        self.landmarks_y.append(y)

    def update_step(self, max_angle, max_distance):

        measurements = []
        for landmark in self.landmarks:
            
            measurement = landmark.make_observation(self.robot.current)
            if (abs(measurement.theta) < max_angle and measurement.r < max_distance): 
                measurements.append(measurement)

        self.ekf.update_step(measurements)

    def prediction_step(self, v, w, dt):

        self.robot.move(v, w, dt)

        v_noise = v + random.gauss(0, self.mu_v)
        w_noise = w + random.gauss(0, self.mu_w)

        self.robot_noise.move(v_noise, w_noise, dt)
        self.ekf.prediction_step(v_noise, w_noise, dt)


    def get_estimate(self):

        return self.ekf.estimate

    def get_covariance(self):

        return self.ekf.covariance

    def get_landmarks_x(self):

        return np.array(self.landmarks_x, ndmin=2)

    def get_landmarks_y(self):

        return np.array(self.landmarks_y, ndmin=2)

    def get_odom(self):

        return np.array(self.robot_noise.current, ndmin=2)

    def get_robot(self):

        return np.array(self.robot.current, ndmin=2)