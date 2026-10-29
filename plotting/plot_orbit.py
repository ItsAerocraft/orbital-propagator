import matplotlib.pyplot as plt


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
