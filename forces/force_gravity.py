import numpy as np


def gravity_accel(t, state, config):
    '''
    Computes the 3D acceleration vector due to gravitational attraction.

    Args:
        [t]: Elapsed time since epoch (s), unused (gravity depends only on position).
        [state]: State vector whose first three elements are the ECI position (m).
        [config]: Simulation config, providing [mu] (m^3 s^-2).

    Returns:
        3-element array representing the gravitational acceleration (m s^-2).

    Standard two-body gravity: a = -[mu] / r^3 * [pos_vctr].
    '''

    pos_vctr = np.array(state[0:3])
    mag_r = np.linalg.norm(pos_vctr)

    grav_factor = - config.mu / (mag_r ** 3)

    return grav_factor * pos_vctr
