import numpy as np

from tumble.inertia import inertia_cuboid, inertia_cylinder

def angular_acceleration(omega, torque, config): #diagonal I matrix only!
    """
    Function takes angular velocity and torque component values (at an instant in time) as the inputs.
    Function computes the components of angular acceleration at that instant in time. 
    Due to the object's axis of rotation aligning with body frame's axis system, using diagonal inertia matrix.
    """

    #extracts angular velocity and torque components
    om_x, om_y, om_z = omega
    tao_x, tao_y, tao_z = torque

    if config.object == "cylinder":

        #sets attributes of object (as preset in config), then takes the diagonal inertial matrix with this given information
        mass, radius, height = config.m, config.radius_cylinder, config.height_cylinder
        I = inertia_cylinder(mass, radius, height)

        I_xx, I_yy, I_zz = I[0, 0], I[1, 1], I[2, 2] #row, column extraction from inertia matrix

    elif config.object == "cuboid":

        #same as above
        mass, length, width, height = config.m, config.length_cuboid, config.width_cuboid, config.height_cuboid
        I = inertia_cuboid(mass, length, width, height)

        I_xx, I_yy, I_zz = I[0, 0], I[1, 1], I[2, 2]

    #subtract gyroscopic term from torque components and multiply by 1 / I 
    om_x_dot = (1 / I_xx) * (tao_x - ((I_zz - I_yy) * om_y * om_z))
    om_y_dot = (1 / I_yy) * (tao_y - ((I_xx - I_zz) * om_x * om_z))
    om_z_dot = (1 / I_zz) * (tao_z - ((I_yy - I_xx) * om_x * om_y))

    return np.array([om_x_dot, om_y_dot, om_z_dot])
