import numpy as np
import random

from base_ekf import Measurement, MotionModel, normalize_angle

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

        # # Add noise
        r     += random.gauss(0, Landmark.mu_r)
        theta += random.gauss(0, Landmark.mu_theta)
        theta  = normalize_angle(theta)

        return Measurement(self.id, abs(r), theta)


class Robot():

    def __init__(self):

        self.current = np.array([0, 0, 0])


    def move(self, v, w, dt):

        self.current  = MotionModel.get_Pose_Prediction(self.current, v, w, dt)