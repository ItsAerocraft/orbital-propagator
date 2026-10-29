import numpy as np

import matplotlib.pyplot as plt

from plotting.plot_style import minorgridlines

from tumble.area import projected_area_cylinder, projected_area_cuboid


def plot_silhouette(t_array, state_array, config):
    '''
    Plots the object's drag-reference silhouette area against time.

    Args:
        [t_array]: Array of sample times (s).
        [state_array]: Array of state vectors, one per row.
        [config]: Simulation config, providing [object] (shape flag).
    '''
    area_array = np.array([
        projected_area_cuboid(state_array[i], config) if config.object == "cuboid"
        else projected_area_cylinder(state_array[i], config)
        for i in range(len(t_array))
    ])

    plt.figure(figsize = (10,6))
    plt.plot(t_array / 86400, area_array)

    minorgridlines()

    plt.xlabel("Time / day", fontsize = 12)
    plt.ylabel("Silhouette Area / m$^2$", fontsize = 12)
    plt.title("Silhouette Area against Time")

    plt.show()
