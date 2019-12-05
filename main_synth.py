from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import random

from ekf import Measurement, EKF_SLAM, MotionModel, normalize_angle
from synth_base import Landmark, Robot
from animations import plot_angle, confidence_ellipse

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