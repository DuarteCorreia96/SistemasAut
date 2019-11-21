import numpy as np
import sys

class MotionModel():

    d = 0.20

    @staticmethod
    def get_Pose_Prediction(current, vl, vr, dt):

        v = (vr + vl) / 2
        w = (vr - vl) / 2 / MotionModel.d
        w_dt = w * dt

        if (w != 0):
            v_w  = v / w
            x = - v_w * np.sin(current[2]) + v_w * np.sin(current[2] + w_dt)
            y =   v_w * np.cos(current[2]) - v_w * np.cos(current[2] + w_dt)

        else:
            x = v * dt * np.cos(current[2])
            y = v * dt * np.sin(current[2])

        return current + np.transpose(np.array([x, y, w_dt])) 

    
    @staticmethod
    def get_Pose_Covariance(current, vl, vr, dt):

        v = (vr + vl) / 2
        w = (vr - vl) / 2 / MotionModel.d
        w_dt = w * dt

        covariance = np.identity(3)

        if (w != 0):
            v_w  = v / w
            covariance[0][2] =   v_w * np.cos(current[2]) + v_w * np.cos(current[2] + w_dt)
            covariance[1][2] = - v_w * np.sin(current[2]) + v_w * np.sin(current[2] + w_dt)

        else:
            covariance[0][2] = dt # This shold probably be some constant ???
            covariance[1][2] = dt

        return covariance

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

class Measurement():

    def __init__(self, id, distance, angle):

        self.id = id
        self.r  = distance
        self.theta = angle


class EKF_SLAM():

    def __init__(self, MotionModel):
 
        self.motion_model = MotionModel
        self.estimate   = np.zeros((3, 1))
        self.covariance = np.zeros((3, 3))

        self.R = np.identity(3) * 0.0
        self.Q = np.diag([0.0, 0.0])
        self.ids = Correspondence()


    def print_state(self):

        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        print('\n\n')
        print('Estimate: \n', self.estimate)
        print('Covariance: \n', self.covariance)

    def add_landmark(self, measure = None):

        self.estimate = np.concatenate((self.estimate, np.zeros((2, 1))))

        if (isinstance(measure, Measurement)):
            self.estimate[-2] = self.estimate[0] + measure.r * np.cos(measure.theta + self.estimate[2])
            self.estimate[-1] = self.estimate[1] + measure.r * np.sin(measure.theta + self.estimate[2])

        max_value = 1000
        self.covariance = np.hstack((self.covariance, max_value * np.ones((self.covariance.shape[0], 2))))
        self.covariance = np.vstack((self.covariance, max_value * np.ones((2, self.covariance.shape[1]))))


    def prediction_step(self, vl, vr, dt):

        # Update estimate of pose
        self.estimate[:3, 0] += self.motion_model.get_Pose_Prediction(self.estimate[:3, 0], vl, vr, dt)

        # Update pose convariance
        jacob = self.motion_model.get_Pose_Covariance(self.estimate[:3], vl, vr, dt)
        self.covariance[ :3,  :3] = jacob.dot(self.covariance[ :3,  :3]).dot(np.transpose(jacob))

        # Update landmarks convariances in relation to pose
        aux = jacob.dot(self.covariance[ :3, 3: ]) 
        self.covariance[ :3, 3: ] = aux
        self.covariance[3: ,  :3] = np.transpose(aux)

        # Add noise
        self.covariance[ :3,  :3] += self.R

    def update_step(self, measurements):

        # This should be modfied after confirming measurements data structure
        sum_estimate    = np.zeros((self.estimate.shape[0], 1))
        sum_convariance = np.zeros(self.covariance.shape)

        for measure in measurements:
            
            j = self.ids.get_index(measure.id)
            if (j == None):

                j = self.ids.add_id(measure.id)
                self.add_landmark(measure)

                sum_estimate = np.vstack((sum_estimate, np.zeros((2, 1))))

                sum_convariance = np.hstack((sum_convariance, np.zeros((sum_convariance.shape[0], 2))))
                sum_convariance = np.vstack((sum_convariance, np.zeros((2, sum_convariance.shape[1]))))

        for measure in measurements:
            
            j = self.ids.get_index(measure.id)

            # Start index of measurement
            j_x = 3 + j * 2
            j_y = 4 + j * 2

            # Calculate deviations to robot
            deviat_x = (self.estimate[j_x] - self.estimate[0])[0]
            deviat_y = (self.estimate[j_y] - self.estimate[1])[0]
            deviat   = np.array([deviat_x, deviat_y])

            sq_error = np.transpose(deviat).dot(deviat)
            error    = np.sqrt(sq_error)

            estimation_x = error
            estimation_y = (np.arctan2(deviat_x, deviat_y) - self.estimate[2])[0]
            estimation   =  np.matrix([[estimation_x], [estimation_y]])

            F_xj = np.zeros((5, self.covariance.shape[1]))
            F_xj[ :3, :3] = np.identity(3) 
            F_xj[3: , j_x:j_y + 1] = np.identity(2)

            H_j_x = np.array([error*deviat_x, -error*deviat_y, 0, -error*deviat_x, error*deviat_y])
            H_j_y = np.array([deviat_y, deviat_x, -1, -deviat_y, -deviat_x])
            H_j   = (1 / sq_error * np.matrix([H_j_x, H_j_y])).dot(F_xj)

            K = self.covariance.dot(np.transpose(H_j))
            K_aux = H_j.dot(self.covariance).dot(np.transpose(H_j)) + self.Q 
            K = K.dot(np.linalg.inv(K_aux))

            measurement = np.matrix([[measure.r], [measure.theta]])
            print(np.transpose(measurement))
            print(np.transpose(estimation))

            sum_estimate    += K.dot(measurement - estimation) 
            sum_convariance += K.dot(H_j)

        self.estimate   = self.estimate + sum_estimate
        self.covariance = (np.identity(self.covariance.shape[0]) - sum_convariance).dot(self.covariance)

     
def main():

    ekf = EKF_SLAM(MotionModel)
    ekf.print_state()

    ekf.print_state()

    ekf.prediction_step(1, 1.1, 1)
    ekf.print_state()

    measurs = [Measurement(1, 2, 0.4), Measurement(2, 2, 0.25)]
    ekf.update_step(measurs)
    ekf.print_state()

    measurs = [Measurement(1, 2, 0.4), Measurement(2, 2, 0.25)]
    ekf.update_step(measurs)
    ekf.print_state()

if __name__ == "__main__":
    main()