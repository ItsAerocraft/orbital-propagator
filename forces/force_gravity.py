import numpy as np

def gravity_accel(t, state, config):
    """
    Function returns 3D acceleration vector for the object due to gravitational attraction.
    """

    pos_vctr = np.array(state[0:3]) # extract items x, y, z from state vector
    mag_r = np.linalg.norm(pos_vctr) # obtain magnitude of position vector

    grav_factor = - config.mu / (mag_r ** 3) # - GM / r^3

    return grav_factor * pos_vctr # grav_factor * each element of the array
