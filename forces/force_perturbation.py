import numpy as np


def J2_accel(t, state, config):
    '''
    Computes the 3D acceleration vector due to J2 perturbation (equatorial oblateness).

    Args:
        [t]: Elapsed time since epoch (s), unused (perturbation is time-independent).
        [state]: State vector whose first three elements are the ECI position (m).
        [config]: Simulation config, providing [J2] (unitless), [mu] (m^3 s^-2),
            and [R] (m).

    Returns:
        3-element array representing the J2 perturbation acceleration (m s^-2).
    '''

    pos_vctr = np.array(state[0:3])
    x, y, z = pos_vctr
    mag_r = np.linalg.norm(pos_vctr)

    J2_factor = - (3 * config.J2 * config.mu * (config.R ** 2)) / (2 * (mag_r ** 5))
    J2_const1 = (5 * (z ** 2) / (mag_r ** 2))

    return np.array([J2_factor * (1 - J2_const1) * x,
                     J2_factor * (1 - J2_const1) * y,
                     J2_factor * (3 - J2_const1) * z
    ])


def J3_accel(t, state, config):
    '''
    Computes the 3D acceleration vector due to J3 perturbation (north-south asymmetry).

    Args:
        [t]: Elapsed time since epoch (s), unused (perturbation is time-independent).
        [state]: State vector whose first three elements are the ECI position (m).
        [config]: Simulation config, providing [J3] (unitless), [mu] (m^3 s^-2),
            and [R] (m).

    Returns:
        3-element array representing the J3 perturbation acceleration (m s^-2).
    '''

    pos_vctr = np.array(state[0:3])
    x, y, z = pos_vctr
    mag_r = np.linalg.norm(pos_vctr)

    J3_factor = - (5 * config.J3 * config.mu * (config.R ** 3)) / (2 * (mag_r ** 9))
    J3_const1 = z * (3 * (x ** 2) + 3 * (y ** 2) - 4 * (z ** 2))
    J3_const2 =  (3 * (x ** 4) + 6 * (x ** 2) * ((y ** 2) - (4 * (z ** 2))) + 3 * (y ** 4) - 24 * ((y * z) ** 2) + 8 * (z ** 4))

    return np.array([J3_factor * x * J3_const1,
                     J3_factor * y * J3_const1,
                     ((J3_factor) / 5) * J3_const2
    ])


def J4_accel(t, state, config):
    '''
    Computes the 3D acceleration vector due to J4 perturbation (symmetric polar flattening).

    Args:
        [t]: Elapsed time since epoch (s), unused (perturbation is time-independent).
        [state]: State vector whose first three elements are the ECI position (m).
        [config]: Simulation config, providing [J4] (unitless), [mu] (m^3 s^-2),
            and [R] (m).

    Returns:
        3-element array representing the J4 perturbation acceleration (m s^-2).
    '''

    pos_vctr = np.array(state[0:3])
    x, y, z = pos_vctr
    mag_r = np.linalg.norm(pos_vctr)

    J4_factor = (15 * config.J4 * config.mu * (config.R ** 4)) / (8 * (mag_r ** 11))
    J4_const1 = ((x ** 4) + 2 * (x ** 2) * ((y ** 2) - (6 * (z ** 2))) + (y ** 4) - 12 * ((y * z) ** 2) + 8 * (z ** 4))
    J4_const2 = (15 * (x ** 4) + 10 * (x ** 2) * ((3 * (y ** 2)) - (4 * (z ** 2))) + (15 * (y ** 4)) - 40 * ((y * z) ** 2) + 8 * (z ** 4))

    return np.array([J4_factor * x * J4_const1,
                     J4_factor * y * J4_const1,
                     ((J4_factor) / 3) * z * J4_const2
    ])


def perturbation_accel(t, state, config):
    '''
    Sums the J2, J3, and J4 perturbation accelerations into one 3D acceleration vector.

    Args:
        [t]: Elapsed time since epoch (s).
        [state]: State vector whose first three elements are the ECI position (m).
        [config]: Simulation config, providing [J2], [J3], [J4] (unitless),
            [mu] (m^3 s^-2), and [R] (m).

    Returns:
        3-element array representing the combined perturbation acceleration (m s^-2).
    '''
    return J2_accel(t, state, config) + J3_accel(t, state, config) + J4_accel(t, state, config)
