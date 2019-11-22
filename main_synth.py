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
    w = 0.1
    dt = 1

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
    ekf = EKF_SLAM(MotionModel)
    ekf.print_state()
    print("Current", robot.current)

    counter = 330
    plt.figure()
    for i in range(8):

        measurements = []
        for landmark in landmarks:
            measurements.append(landmark.make_observation(robot.current))

        # for measurement in measurements:
        #     print(str(measurement))

        ekf.update_step(measurements)

        counter += 1
        plt.subplot(counter)

        plt.plot(robot.current[0], robot.current[1], 'gs')
        plt.plot(ekf.estimate[0] , ekf.estimate[1] , 'ro', markersize=2)

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
        plt.axis([-1, 5, -1, 5])

        plt.title('Estimation at time:' + str(i))

        robot.move(v, w, dt)
        ekf.prediction_step(v, w, dt)

        ekf.print_state()
        print("Current", robot.current)


    plt.subplot(counter + 1)
    plt.axis('off')
    from matplotlib.lines import Line2D
    legend_elements = [ Line2D([0], [0], marker='^', color='g', label='Landmarks'),
                        Line2D([0], [0], marker='x', color='r', label='Landmarks_est'),
                        Line2D([0], [0], marker='s', color='g', label='Robot'),
                        Line2D([0], [0], marker='o', color='r', label='Robot_est')
                    ]
    plt.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()