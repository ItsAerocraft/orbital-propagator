import numpy as np

import matplotlib.pyplot as plt

from plotting.plot_style import minorgridlines


def plot_altitude(t_array, state_array, config):
    '''
    Plots altitude above ground level against time.

    Args:
        [t_array]: Array of sample times (s).
        [state_array]: Array of state vectors, one per row, whose columns
            0:3 are the ECI position (m).
        [config]: Simulation config, providing [R] (m).
    '''
    altitude_array = np.linalg.norm(state_array[ : , 0:3], axis = 1) - config.R

    plt.figure(figsize = (10, 6))
    plt.plot(t_array / 86400, altitude_array / 1000)

    minorgridlines()

    plt.xlabel("Time / day", fontsize = 12)
    plt.ylabel("Altitude / km", fontsize = 12)
    plt.title("Altitude against Time", fontsize = 14)

    plt.show()
