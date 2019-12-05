import numpy as np
import random

from ekf import Measurement, EKF_SLAM, MotionModel, normalize_angle

class Landmark():

    mu_r     = 0
    mu_theta = 0

    def __init__(self, mark_id, x, y):

        self.id = mark_id
        self.x  = x
        self.y  = y

    def make_observation(self, current):

        deviat_x = self.x - current[0]
        deviat_y = self.y - current[1]

        r     = np.sqrt(deviat_x ** 2 + deviat_y ** 2)     
        theta = np.arctan2(deviat_y, deviat_x) - current[2]
        theta = normalize_angle(theta)

        # # Add noise
        r     += random.gauss(0, Landmark.mu_r)
        theta += random.gauss(0, Landmark.mu_theta)

        return Measurement(self.id, r, theta)


class Robot():

    def __init__(self):

        self.current = np.array([0, 0, 0])


    def move(self, v, w, dt):

        self.current  = MotionModel.get_Pose_Prediction(self.current, v, w, dt)


class Matlab_EKF():

    def __init__(self, Q_diag, sigma, mu_odom, mu_observation):

        Q = np.diag(Q_diag)
        MotionModel.sigma = sigma

        # Measurement noises
        self.mu_v = mu_odom[0]
        self.mu_w = mu_odom[1]

        Landmark.mu_r     = mu_observation[0]
        Landmark.mu_theta = mu_observation[1]

        self.landmarks = []

        self.robot = Robot()
        self.robot_noise = Robot()
        self.ekf = EKF_SLAM(MotionModel, Q = Q)

        self.landmarks_x = []
        self.landmarks_y = []

    def add_landmark(self, mark_id, x, y):

        self.landmarks.append(Landmark(mark_id, x, y))
        self.landmarks_x.append(x)
        self.landmarks_y.append(y)

    def update_step(self):

        measurements = []
        for landmark in self.landmarks:
            measurements.append(landmark.make_observation(self.robot.current))

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
