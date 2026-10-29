import numpy as np

import matplotlib.pyplot as plt

from plotting.style import minorgridlines

from tumble.area import projected_area_cylinder, projected_area_cuboid


def plot_3D(x_array, y_array, z_array):
    '''
    Plots the 3D orbital trajectory.

    Args:
        [x_array]: Array of ECI x-positions (m).
        [y_array]: Array of ECI y-positions (m).
        [z_array]: Array of ECI z-positions (m).
    '''
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111, projection = "3d")

    x_range = x_array.max() - x_array.min() + 1
    y_range = y_array.max() - y_array.min() + 1
    z_range = z_array.max() - z_array.min() + 1

    ax.set_box_aspect([x_range, y_range, z_range])

    ax.plot(x_array, y_array, z_array)

    ax.scatter([0], [0], [0], color = "red", s = 50, label = "Earth center")
    ax.legend()

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")

    plt.show()


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


def plot_single_run(t_array, state_array, config):
    '''
    Plots every graph for a single run: the 3D orbit, altitude, and silhouette area.

    Args:
        [t_array]: Array of sample times (s).
        [state_array]: Array of state vectors, one per row.
        [config]: Simulation config.
    '''
    plot_3D(state_array[ : , 0], state_array[ : , 1], state_array[ : , 2])
    plot_altitude(t_array, state_array, config)
    plot_silhouette(t_array, state_array, config)
