import matplotlib.transforms as transforms
import matplotlib.animation as manimation
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import random

from ekf import Measurement, EKF_SLAM, MotionModel, normalize_angle

def plot_angle(estimate, style = 'b'):

    x_line = [estimate[0].item(), estimate[0].item() + 0.5 * np.cos(estimate[2].item())]
    y_line = [estimate[1].item(), estimate[1].item() + 0.5 * np.sin(estimate[2].item())]

    plt.plot(x_line, y_line, style)

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


def confidence_ellipse(x, y, ax, cov, facecolor='none', style = 'rx',**kwargs):

    pearson = cov[0, 1] / (cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0),
        width  = ell_radius_x * 2,
        height = ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0])
    mean_x  = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1])
    mean_y  = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    plt.plot(x,y, style, markersize = 4)

    return ax.add_patch(ellipse)

def main():

    # EKF parameters
    Q = np.diag([5, 5])
    MotionModel.sigma = 0.03

    # Measurement noises
    mu_v = 0.2
    mu_w = 0.05

    Landmark.mu_r     = 0.5
    Landmark.mu_theta = 0.02

    # Video parameters and simulation
    v = 0.2
    w = 0.02
    fps = 60
    dt  = 1 / fps
    frames  = 250
    skipped = 10
    steps   = frames * skipped
    _print  = False

    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer   = manimation.FFMpegWriter(fps = fps, metadata = metadata)

    landmarks = []
    landmarks.append(Landmark( 1, 2,  2))
    landmarks.append(Landmark( 2, 3,  3))
    landmarks.append(Landmark( 3, 0,  4))
    landmarks.append(Landmark( 4, 2,  4))

    robot = Robot()
    robot_noise = Robot()
    ekf = EKF_SLAM(MotionModel, Q = Q)

    if (_print == True):
        ekf.print_state()
        print("Current", robot.current)

    plot_step = steps // 8 
    if(plot_step - (steps / 8) != 0):
        plot_step += 1
    
    landmarks_x = []
    landmarks_y = []
    for landmark in landmarks:
        landmarks_x.append(landmark.x)
        landmarks_y.append(landmark.y)

    robot_path_x = [0]
    robot_path_y = [0]

    estim_path_x = [0]
    estim_path_y = [0]

    noise_path_x = [0]
    noise_path_y = [0]

    fig, covariance_plot = plt.subplots(figsize=(6, 6))
    with writer.saving(fig, "writer_test.mp4", steps / skipped):
        
        for i in range(steps):

            measurements = []
            for landmark in landmarks:
                measurements.append(landmark.make_observation(robot.current))

            # for measurement in measurements:
            #     print(str(measurement))

            ekf.update_step(measurements)

            if (i % skipped == 0):

                print("Frame:", i / skipped)
                covariance_plot.clear()

                robot_path_x.append(robot.current[0])
                robot_path_y.append(robot.current[1])

                estim_path_x.append(ekf.estimate[0].item())
                estim_path_y.append(ekf.estimate[1].item())

                plt.xlim(-1, 10)
                plt.ylim(-1, 10)

                confidence_ellipse(
                    ekf.estimate[0].item(),
                    ekf.estimate[1].item(),
                    covariance_plot, 
                    ekf.covariance[:2,:2],
                    style='co',
                    edgecolor='cyan'
                )

                noise_path_x.append(robot_noise.current[0])
                noise_path_y.append(robot_noise.current[1])

                robot_vid, = plt.plot([], [], 'g-s', markersize = 2)
                odome_vid, = plt.plot([], [], 'r-s', markersize = 2)
                estim_vid, = plt.plot([], [], 'c-o', markersize = 2)
                
                robot_vid.set_data(robot_path_x, robot_path_y)
                odome_vid.set_data(noise_path_x, noise_path_y)
                estim_vid.set_data(estim_path_x, estim_path_y)
    
                plt.plot(landmarks_x, landmarks_y, 'g^')
                plt.grid(True)
                plt.xlabel('x')
                plt.ylabel('y')

                j = 0
                for _ in landmarks:
                    x  = 3 + j * 2
                    y  = 4 + j * 2
                    confidence_ellipse(
                        ekf.estimate[x].item(),
                        ekf.estimate[y].item(),
                        covariance_plot, 
                        ekf.covariance[x:y+1,x:y+1],
                        edgecolor='red'
                    )

                    j += 1

                writer.grab_frame()

            robot.move(v, w, dt)

            v_noise = v + random.gauss(0, mu_v)
            w_noise = w + random.gauss(0, mu_w)

            robot_noise.move(v_noise, w_noise, dt)
            ekf.prediction_step(v_noise, w_noise, dt)

            if (_print == True):
                ekf.print_state()
                print("Current", robot.current)


    ekf.print_state()

if __name__ == "__main__":
    main()