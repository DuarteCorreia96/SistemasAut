import matplotlib.pyplot as plt
import numpy as np
import random

from ekf import Measurement, EKF_SLAM, MotionModel, normalize_angle

class Landmark():

    def __init__(self, mark_id, x, y):

        self.id = mark_id
        self.x  = x
        self.y  = y

    def make_observation(self, current):

        mu_r     = 0.1
        mu_theta = 0.02
        deviat_x = self.x - current[0]
        deviat_y = self.y - current[1]

        r     = np.sqrt(deviat_x**2 + deviat_y**2)          + random.gauss(0, mu_r)
        theta = np.arctan2(deviat_y, deviat_x) - current[2] + random.gauss(0, mu_theta)

        return Measurement(self.id, r, theta)


class Robot():

    def __init__(self):

        self.current = np.array([0, 0, 0])


    def move(self, v, w, dt):

        self.current  = MotionModel.get_Pose_Prediction(self.current, v, w, dt)


def main():

    v = 0.5
    w = 0.15
    dt = 0.01
    steps = 1000
    _print = False

    mu_v = 0.5
    mu_w = 0.5

    landmarks = []
    landmarks.append(Landmark(1, 2,  2))
    landmarks.append(Landmark(2, 3,  3))
    landmarks.append(Landmark(3, 0,  4))
    landmarks.append(Landmark(4, 2,  4))

    landmarks_x = []
    landmarks_y = []
    for landmark in landmarks:
        landmarks_x.append(landmark.x)
        landmarks_y.append(landmark.y)

    robot = Robot()
    robot_noise = Robot()
    ekf = EKF_SLAM(MotionModel)

    if (_print == True):
        ekf.print_state()
        print("Current", robot.current)

    plot_step = steps // 8 
    if(plot_step - (steps / 8) != 0):
        plot_step += 1

    counter = 330
    plt.figure()

    robot_path_x = [0]
    robot_path_y = [0]

    estim_path_x = [0]
    estim_path_y = [0]

    noise_path_x = [0]
    noise_path_y = [0]
    for i in range(steps):

        measurements = []
        for landmark in landmarks:
            measurements.append(landmark.make_observation(robot.current))

        # for measurement in measurements:
        #     print(str(measurement))

        ekf.update_step(measurements)

        if (i % plot_step == 0):
            counter += 1
            plt.subplot(counter)

            plt.plot(robot.current[0], robot.current[1], 'gs')
            plt.plot(robot_noise.current[0], robot_noise.current[1], 'rs')
            plt.plot(ekf.estimate[0], ekf.estimate[1] , 'yo', markersize=2)

            robot_path_x.append(robot.current[0])
            robot_path_y.append(robot.current[1])

            estim_path_x.append(ekf.estimate[0])
            estim_path_y.append(ekf.estimate[1])

            noise_path_x.append(robot_noise.current[0])
            noise_path_y.append(robot_noise.current[1])

            plt.plot(robot_path_x, robot_path_y, 'g')
            plt.plot(estim_path_x, estim_path_y, 'y')
            plt.plot(noise_path_x, noise_path_y, 'r')

            plt.plot(landmarks_x, landmarks_y, 'g^')

            j = 0
            for _ in landmarks:
                x  = 3 + j * 2
                y  = 4 + j * 2
                j += 1
                plt.plot(ekf.estimate[x], ekf.estimate[y], 'rx')

            plt.grid(True)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis([-1, 10, -1, 6])

            plt.title('Estimation at time: ' + str(np.round(i * dt, decimals=2)) + " sec")

        robot.move(v, w, dt)

        v_noise = v + random.gauss(0, mu_v)
        w_noise = w + random.gauss(0, mu_w)

        robot_noise.move(v_noise, w_noise, dt)
        ekf.prediction_step(v_noise, w_noise, dt)

        if (_print == True):
            ekf.print_state()
            print("Current", robot.current)


    plt.subplot(counter + 1)
    plt.axis('off')
    from matplotlib.lines import Line2D
    legend_elements = [ Line2D([0], [0], marker='^', color='g', label='Landmarks'),
                        Line2D([0], [0], marker='x', color='r', label='Landmarks_est'),
                        Line2D([0], [0], marker='s', color='g', label='Robot'),
                        Line2D([0], [0], marker='o', color='y', label='Robot_est'),
                        Line2D([0], [0], marker='s', color='r', label='Robot_Odom')
                    ]
    plt.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()