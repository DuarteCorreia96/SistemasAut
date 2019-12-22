import numpy as np

def normalize_angle(angle):

    angle = angle % (2 * np.pi)
    if (angle >= np.pi):
        angle -= 2 * np.pi

    return angle

class MotionModel():

    L = 100
    sigma = 0.05

    @staticmethod
    def get_Pose_Prediction(current, v, w, dt):

        x     = v * np.cos(current[2]) * dt
        y     = v * np.sin(current[2]) * dt
        theta = w * dt
        
        updated = current + np.array([x, y, theta])

        return updated

    @staticmethod
    def get_Pose_Covariance(current, current_cov, v, dt):

        Gx = np.identity(3)
        c  = np.asscalar(np.cos(current[2]))
        s  = np.asscalar(np.sin(current[2]))

        B = np.zeros((3,3))
        B[0, 2] = -s * v * dt
        B[1, 2] =  c * v * dt
        Gx += B

        Ge = np.matrix([[c,c], [s,s], [1 / MotionModel.L, -1 / MotionModel.L]])

        R = (MotionModel.sigma ** 2) * np.matmul(Ge, Ge.T)

        return [Gx, R]


class Measurement():

    def __init__(self, mark_id, distance, angle):

        self.id = mark_id
        self.r  = distance
        self.theta = angle

    def __str__(self):
        
        to_print  = "  id: " + str(self.id)
        to_print += "\t r: " + str(self.r)
        to_print += "\t theta: " + str(self.theta)

        return to_print