import numpy as np

from tumble.inertia import inertia_cuboid, inertia_cylinder


def angular_acceleration(omega, torque, config):
    '''
    Computes the angular acceleration components given angular velocity and torque.

    Args:
        [omega]: 3-element body-frame angular velocity vector (rad s^-1).
        [torque]: 3-element body-frame torque vector (N m).
        [config]: Simulation config, providing [object] (shape flag), [m]
            (mass, kg), and the relevant dimensions: [radius_cylinder] (m),
            [height_cylinder] (m) for a cylinder, or [length_cuboid] (m),
            [width_cuboid] (m), [height_cuboid] (m) for a cuboid.

    Returns:
        3-element array representing the angular acceleration (rad s^-2).

    Assumes the body frame's axes are aligned with the object's principal
    axes of rotation, so only the diagonal moment of inertia matrix is used;
    the gyroscopic term is subtracted from each torque component.
    '''

    om_x, om_y, om_z = omega
    tao_x, tao_y, tao_z = torque

    if config.object == "cylinder":
        mass, radius, height = config.m, config.radius_cylinder, config.height_cylinder
        I = inertia_cylinder(mass, radius, height)

        I_xx, I_yy, I_zz = I[0, 0], I[1, 1], I[2, 2]

    elif config.object == "cuboid":
        mass, length, width, height = config.m, config.length_cuboid, config.width_cuboid, config.height_cuboid
        I = inertia_cuboid(mass, length, width, height)

        I_xx, I_yy, I_zz = I[0, 0], I[1, 1], I[2, 2]

    om_x_dot = (1 / I_xx) * (tao_x - ((I_zz - I_yy) * om_y * om_z))
    om_y_dot = (1 / I_yy) * (tao_y - ((I_xx - I_zz) * om_x * om_z))
    om_z_dot = (1 / I_zz) * (tao_z - ((I_yy - I_xx) * om_x * om_y))

    return np.array([om_x_dot, om_y_dot, om_z_dot])
