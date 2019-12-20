import numpy as np
import sys

from base_ekf import Measurement, MotionModel, normalize_angle

class Correspondence():
    """ This class is just to save the already found marers ids 
    and cna be expanded to no know correspondences
    """

    def __init__(self):

        self.id_counter = 0
        self.ids = {}

    def add_id(self, marker_id):

        self.ids[marker_id] = self.id_counter
        self.id_counter += 1

        return self.ids[marker_id]

    def get_index(self, marker_id):

        if (marker_id in self.ids):
            return self.ids[marker_id]
        else:
            return None

class EKF_SLAM():

    def __init__(self, MotionModel, Q = np.diag([0.1, 0.01])):
 
        self.motion_model = MotionModel
        self.estimate   = np.zeros((3, 1))
        self.covariance = np.zeros((3, 3))

        self.ids = Correspondence()

        # Tunning factors of the EKF
        self.Q = Q

        # Value to initialize convariance of landmarks
        self.max_conv = 10000000

    def print_state(self):

        np.set_printoptions(precision = 3)
        np.set_printoptions(suppress  = True)
        np.set_printoptions(linewidth = 200)

        print('\n\n')
        print('Estimate^T: \n', np.transpose(self.estimate))
        print('Covariance: \n', self.covariance)

    def add_landmark(self, measure = None):

        self.estimate = np.concatenate((self.estimate, np.zeros((2, 1))))

        if (isinstance(measure, Measurement)):
            self.estimate[-2] = self.estimate[0] + measure.r * np.cos(measure.theta + self.estimate[2])
            self.estimate[-1] = self.estimate[1] + measure.r * np.sin(measure.theta + self.estimate[2])

        self.covariance = np.hstack((self.covariance, np.zeros((self.covariance.shape[0], 2))))
        self.covariance = np.vstack((self.covariance, np.zeros((2, self.covariance.shape[1]))))

        self.covariance[-1, -1] = self.max_conv
        self.covariance[-2, -2] = self.max_conv

    def prediction_step(self, v, w, dt):

        # Update estimate of pose
        self.estimate[:3, 0] = self.motion_model.get_Pose_Prediction(self.estimate[:3, 0], v, w, dt)

        # Update pose convariance
        [jacob, R] = self.motion_model.get_Pose_Covariance(self.estimate[:3], self.covariance[ :3,  :3], v, dt)
        self.covariance[ :3,  :3] = jacob.dot(self.covariance[ :3,  :3]).dot(np.transpose(jacob))

        # Update landmarks convariances in relation to pose
        aux = jacob.dot(self.covariance[ :3, 3: ]) 
        self.covariance[ :3, 3: ] = aux
        self.covariance[3: ,  :3] = np.transpose(aux)

        # Add noise
        self.covariance[ :3,  :3] += R

    def update_step(self, measurements):

        # This should be modfied after confirming measurements data structure
        for measure in measurements:
            j = self.ids.get_index(measure.id)
            if (j == None):

                j = self.ids.add_id(measure.id)
                self.add_landmark(measure)

        sum_estimate    = np.zeros((self.estimate.shape[0], 1))
        sum_convariance = np.zeros(self.covariance.shape)

        for measure in measurements:
            
            j = self.ids.get_index(measure.id)

            # Start index of measurement
            j_x = 3 + j * 2
            j_y = 4 + j * 2

            # Calculate deviations to robot
            deviat_x = np.asscalar(self.estimate[j_x] - self.estimate[0])
            deviat_y = np.asscalar(self.estimate[j_y] - self.estimate[1])
            deviat   = np.matrix([[deviat_x], [deviat_y]])

            sq_error = np.asscalar(np.transpose(deviat).dot(deviat))
            error    = np.sqrt(sq_error)

            if (sq_error != 0):

                F_xj = np.zeros((5, self.covariance.shape[1]))
                F_xj[ :3, :3] = np.identity(3) 
                F_xj[3: , j_x:j_y + 1] = np.identity(2)

                H_x = np.array([-error*deviat_x, -error*deviat_y, 0, error*deviat_x, error*deviat_y]) / sq_error
                H_y = np.array([deviat_y, -deviat_x, -sq_error, -deviat_y, deviat_x]) / sq_error
                H   = np.matrix([H_x, H_y]).dot(F_xj)

                K     = self.covariance.dot(np.transpose(H))
                K_aux = np.add(H.dot(self.covariance).dot(np.transpose(H)), self.Q * (measure.r ** 2)) 
                K     = K.dot(np.linalg.inv(K_aux))

                estimation_r     = error
                estimation_theta = np.asscalar(np.arctan2(deviat_y, deviat_x) - self.estimate[2])
                estimation_theta = normalize_angle(estimation_theta)

                measure_dev      = np.matrix([[measure.r - estimation_r], [normalize_angle(measure.theta - estimation_theta)]])
                sum_estimate    += K.dot(measure_dev) 
                sum_convariance += K.dot(H)


        self.estimate    = np.add(self.estimate, sum_estimate)
        self.covariance  = (np.identity(self.covariance.shape[0]) - sum_convariance).dot(self.covariance)
