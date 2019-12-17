import numpy as np
import sys
import copy

from base_ekf import Measurement, MotionModel, normalize_angle

class EKF_SLAM_unknown():

    def __init__(self, MotionModel, Q = np.diag([0.1, 0.01]), alpha = 50):
 
        self.motion_model = MotionModel
        self.estimate   = np.zeros((3, 1))
        self.covariance = np.zeros((3, 3))

        # Tunning factors of the EKF
        self.Q = Q

        # Value to initialize convariance of landmarks
        self.max_conv = 10000000

        # Number of landmarks
        self.N = 0

        # Controls new landmark creation
        self.alpha = alpha
        self.update_counter = 0

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

        if (self.N == 0):
            for measure in measurements:
                print("Found new landmark")
                self.add_landmark(measure)
                self.N += 1
            
            return

        for measure in measurements:
            
            best_supp = Supposition(self, 0, measure)
            for j in range(1, self.N):
    
                supp = Supposition(self, j, measure)
                if (supp.pi < best_supp.pi):
                    best_supp = copy.deepcopy(supp)

            if (best_supp.pi < self.alpha):
                if (best_supp.error != 0):
                    K = self.covariance.dot(np.transpose(best_supp.H)).dot(np.linalg.inv(best_supp.psi)) 
                    self.estimate   +=  K.dot(best_supp.measure_dev)
                    self.covariance  = (np.identity(self.covariance.shape[0]) - K.dot(best_supp.H)).dot(self.covariance)

            else:
                print("Found new landmark")
                self.add_landmark(measure)
                self.N += 1

        # This should improve filter by a lot it simply deletes markers with a single observation every 50 observations
        self.update_counter += 1
        if (self.update_counter % 50 == 0):

            delete_counter = 0
            truth = self.covariance == self.max_conv
            for k in range(3, len(truth), 2):
                if (OP(truth[k]) and OP(truth[k + 1])):

                    # delete rows
                    self.covariance = np.delete(self.covariance, k - delete_counter, 0)
                    self.covariance = np.delete(self.covariance, k - delete_counter, 0)

                    self.estimate  = np.delete(self.estimate, k - delete_counter, 0)
                    self.estimate  = np.delete(self.estimate, k - delete_counter, 0)

                    # delete columns
                    self.covariance = np.delete(self.covariance, k - delete_counter, 1)
                    self.covariance = np.delete(self.covariance, k - delete_counter, 1)

                    # update delete counter
                    delete_counter += 2
                    self.N -= 1
            
            self.update_counter = 0

def OP(l):
    true_found = False
    for v in l:
        if v and not true_found:
            true_found=True
        elif v and true_found:
             return False #"Too Many Trues"
    return true_found

class Supposition():

    def __init__(self, ekf, k, measure):

        # Start index of measurement
        j_x = 3 + k * 2
        j_y = 4 + k * 2

        deviat_x = np.asscalar(ekf.estimate[j_x] - ekf.estimate[0])
        deviat_y = np.asscalar(ekf.estimate[j_y] - ekf.estimate[1])
        deviat   = np.array([[deviat_x], [deviat_y]])

        sq_error   = np.asscalar(np.transpose(deviat).dot(deviat))
        error = np.sqrt(sq_error)

        self.error = error
        if (error != 0):

            F_xj = np.zeros((5, ekf.covariance.shape[1]))
            F_xj[ :3, :3] = np.identity(3) 
            F_xj[3: , j_x:j_y + 1] = np.identity(2)

            H_x = np.array([-error*deviat_x, -error*deviat_y, 0, error*deviat_x, error*deviat_y]) / sq_error
            H_y = np.array([deviat_y, -deviat_x, -sq_error, -deviat_y, deviat_x]) / sq_error

            estimation_r     = error
            estimation_theta = np.asscalar(np.arctan2(deviat_y, deviat_x) - ekf.estimate[2])
            estimation_theta = normalize_angle(estimation_theta)

            self.H            = np.array([H_x, H_y]).dot(F_xj)
            self.measure_dev  = np.array([[measure.r - estimation_r], [normalize_angle(measure.theta - estimation_theta)]])
            self.psi          = self.H.dot(ekf.covariance).dot(np.transpose(self.H)) + ekf.Q
            self.pi           = np.asscalar(np.transpose(self.measure_dev).dot(self.psi).dot(self.measure_dev))