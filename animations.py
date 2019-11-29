import matplotlib.transforms as transforms
import matplotlib.animation as manimation
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

def confidence_ellipse(x, y, ax, cov, facecolor='none', **kwargs):

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
    plt.plot(x,y, 'rx', markersize = 4)

    return ax.add_patch(ellipse)

def animate(frame, *fargs):

    robot = fargs[0]
    noise = fargs[1]
    ekf   = fargs[2]

    landmarks   = fargs[3]
    landmarks_x = fargs[4]
    landmarks_y = fargs[5]

    estim_path_x = fargs[6]
    estim_path_y = fargs[7]

    frame.clear()
    plt.xlim(-1, 10)
    plt.ylim(-1, 10)

    confidence_ellipse(
        ekf.estimate[0].item(),
        ekf.estimate[1].item(),
        frame, 
        ekf.covariance[:2,:2],
        edgecolor='yellow'
    )

    robot_vid, = plt.plot([], [], 'g-s', markersize = 2)
    odome_vid, = plt.plot([], [], 'r-s', markersize = 2)
    estim_vid, = plt.plot([], [], 'y-o', markersize = 2)
    
    robot_vid.set_data(robot.path_x, robot.path_y)
    odome_vid.set_data(estim_path_x, estim_path_y)
    estim_vid.set_data(noise.path_x, noise.path_y)

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
            frame, 
            ekf.covariance[x:y+1,x:y+1],
            edgecolor='red'
        )

        j += 1

    return frame