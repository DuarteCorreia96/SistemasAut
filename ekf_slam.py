#! /usr/bin/env python
import numpy as np
import math as m
'''
Motion model
'''
class MotionModel(object):
    def __init__(self, x_t):
        self.x_t = x_t  #x_t = [x, y, theta]^T
        self.R_t = np.zeros(3) #process noise covariance matriz - zeros just for now

    def motionUpdate(self, v_t, w_t, delta_t):
        #Compute coordinate updates
        x_update = -v_t/w_t * np.sin(self.x_t[2]) + v_t/w_t * np.sin(self.x_t[2] + w_t*delta_t)
        y_update = v_t/w_t * np.cos(self.x_t[2]) - v_t/w_t * np.cos(self.x_t[2] + w_t*delta_t) 
        theta_update = w_t*delta_t 
        #Sum the updates to the previous state
        self.x_t += np.array([x_update, y_update, theta_update])
        #return just the update (will be useful for ekf slam)
        return np.array([x_update, y_update, theta_update])

    def low_dim_jacobian(self, v_t, w_t, delta_t):
        #Compute coordinate updates
        j_1 = -v_t/w_t * np.cos(self.x_t[2]) + v_t/w_t * np.cos(self.x_t[2] + w_t*delta_t)
        j_2 = -v_t/w_t * np.sin(self.x_t[2]) + v_t/w_t * np.sin(self.x_t[2] + w_t*delta_t)
        J = np.zeros((3,3))
        J[0,2] = j_1
        J[1,2] = j_2
        return J

'''
Measurements model
'''
class MeasurementsModel(object):
    def __init__(self, Q_t):
        self.Q_t = Q_t
        self.data = None

    def callback(self, data):
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", str(data.markers[0].pose.pose.position.x))
        self.data = data

def convert_pose(pose_x, pose_y):
    r = m.sqrt(m.pow(pose_x,2) + m.pow(pose_y,2))
    angle = m.atan2(pose_y, pose_y)
    return [r, angle]

def estimate_measurement(landmark_x, landmark_y, robot_x, robot_y, robot_angle):
    r_squared = m.pow((landmark_x-robot_x), 2) + m.pow((landmark_y-robot_y), 2)
    r_estimate = m.sqrt(r_squared)
    angle_estimate = m.atan2((landmark_x-robot_x), (landmark_y-robot_y)) - robot_angle
    return [r_estimate, angle_estimate]


'''
EKF SLAM
'''
class EKFSLAM_Problem(object):
    
    def __init__(self, mu, Sigma, model_init_pos, Q_t):
        self.mu = mu
        self.Sigma = Sigma
        self.z = [] #measurements
        self.u = [] #[v_x, v_y, w_z] or [v_x, v_y, v_z, w_x, w_y, w_z]
        self.c = [] #list of seen landmarks (markers)
        self.F_x = np.concatenate( (np.identity(3), np.zeros( shape=(3*len(self.mu[3:], 3))  )) ).T
        self.motion_model = MotionModel(np.array(model_init_pos))
        self.measurements_model = MeasurementsModel(Q_t)
        self.z_hat = np.zeros(len(self.mu))
        self.z = np.zeros(len(self.mu))


    def ekf_slam_known_correspondences(self, delta_t): 
        # 
        
        # motion update
        v_t = np.sqrt(np.power(self.u[0], 2)+np.power(self.u[1], 2))
        w_t = self.u[-1]
        mu_bar = self.mu + np.matmul(self.F_x.T, self.motion_model.motionUpdate(v_t, w_t, delta_t)) #motionUpdate might need to have delta_t as argument if not fixed

        G_t = np.identity(3) + np.matmul(np.matmul(self.F_x.T, self.motion_model.low_dim_jacobian(v_t, w_t, delta_t)), self.F_x)

        Sigma_bar = G_t*self.Sigma*G_t.T + self.F_x.T*self.motion_model.R_t*self.F_x

        for i, landmark in enumerate(self.z):
            #check if the landmark is new
            if landmark.id not in self.c:  #check field
                #append to the c list
                self.c.append(landmark.id)
                j = self.c.index(landmark.id)
                #compute landmark coordinates and append to mu
                mu_bar[3*j] = mu_bar[0] + landmark.r * np.cos(landmark.angle + mu_bar[2])
                mu_bar[3*j+1] = mu_bar[1] + landmark.r * np.sin(landmark.angle + mu_bar[2])
                mu_bar[3*j+2] = landmark.id

            j = self.c.index(landmark.id)

            delta_x = mu_bar[3*j] - mu_bar[0]
            delta_y = mu_bar[3*j+1] - mu_bar[1]

            delta = np.array([delta_x, delta_y])

            q = np.matmul(delta.T, delta)   

            aux = estimate_measurement(mu_bar[3*j], mu_bar[3*j+1], mu_bar[0], mu_bar[1], mu_bar[2])
            self.z_hat[3*j] = aux[0]
            self.z_hat[3*j+1] = aux[1]
            self.z_hat[3*j+2] = landmark.id

            F = np.zeros(shape=(3 + len(self.mu), 3 + len(self.mu)))

            F[0, 0] = 1
            F[1, 1] = 1
            F[2, 2] = 1
            F[3*j, 3*j] = 1
            F[3*j+1, 3*j+1] = 1
            F[3*j+2, 3*j+2] = 1

            H = (1/q) * np.matmul()

            #compute distance to landmark in x and y

            #square euclidean distance



'''
For testing
'''
'''
def main():
    foo = Measure()

    rospy.init_node('aruco_pose_listener', anonymous=True)
    rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, foo.callback)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():

        if foo.data is not None:
            #print(str(foo.data))
            print 'X = '+str(foo.data.markers[0].pose.pose.position.x)+'\n'
            print 'Y = '+str(foo.data.markers[0].pose.pose.position.z)+'\n'
            ll = convert_pose(foo.data.markers[0].pose.pose.position.x, foo.data.markers[0].pose.pose.position.z)
            print 'R = '+str(ll[0])+' Phi = '+str(ll[1])+'\n'
            ld_x = 0 #foo.data.markers[0].pose.pose.position.x
            ld_y = 0 #foo.data.markers[0].pose.pose.position.z
            var = input('Yes or No:' )
            if var == 1:

                r_x = foo.data.markers[0].pose.pose.position.x
                r_y = foo.data.markers[0].pose.pose.position.z 
                r_a = 0
                ll2 = estimate_measurement(ld_x, ld_y, r_x, r_y, r_a)
                
                print 'Robot y ' + str(r_y)
                print 'R estimate ' + str(ll2[0])
                print 'Phi estimate ' + str(ll2[1])
                

        rate.sleep()

main()
'''