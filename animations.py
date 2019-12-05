from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def plot_angle(estimate, style = 'b'):

    x_line = [estimate[0].item(), estimate[0].item() + 0.5 * np.cos(estimate[2].item())]
    y_line = [estimate[1].item(), estimate[1].item() + 0.5 * np.sin(estimate[2].item())]

    plt.plot(x_line, y_line, style)

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