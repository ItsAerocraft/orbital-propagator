import numpy as np


def R_x(angle):
    '''
    Generates the 3x3 rotation matrix for a rotation about the x-axis.

    Args:
        [angle]: Rotation angle (rad).

    Returns:
        3x3 array representing the rotation matrix (unitless).
    '''

    matrix_rotate_x = np.array([[1, 0, 0],
                   [0, np.cos(angle), - np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]
                   ])

    return matrix_rotate_x


def R_z(angle):
    '''
    Generates the 3x3 rotation matrix for a rotation about the z-axis.

    Args:
        [angle]: Rotation angle (rad).

    Returns:
        3x3 array representing the rotation matrix (unitless).
    '''

    matrix_rotate_z = np.array([[np.cos(angle), - np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]
                   ])

    return matrix_rotate_z


def coe_to_cartesian(a, e, theta, omega, inc, raan, config):
    '''
    Converts classical orbital elements into a Cartesian ECI position/velocity state.

    Args:
        [a]: Semi-major axis (m).
        [e]: Eccentricity (unitless).
        [theta]: True anomaly (rad).
        [omega]: Argument of periapsis (rad).
        [inc]: Inclination angle (rad).
        [raan]: Right ascension of the ascending node (rad).
        [config]: Simulation config, providing [mu] (m^3 s^-2).

    Returns:
        6-element array: ECI position (m) concatenated with ECI velocity (m s^-1).

    Computes the perifocal position/velocity vectors (with the perifocal
    x-axis aligned with the vernal equinox and z = 0), then rotates them into
    the ECI frame via R_z([raan]) @ R_x([inc]) @ R_z([omega]).
    '''

    p = a * (1 - (e ** 2))
    h = np.sqrt(config.mu * p)

    r = p / (1 + e * np.cos(theta))

    r_pf = np.array([r * np.cos(theta), r * np.sin(theta), 0])

    v_pf = np.array([ - (h / p) * np.sin(theta), (h / p) * (e + np.cos(theta)), 0])

    rotate_matrix = R_z(raan) @ R_x(inc) @ R_z(omega)

    r_final = rotate_matrix @ r_pf
    v_final = rotate_matrix @ v_pf

    return np.concatenate([r_final, v_final])


def current_semimajor_axis(state, config):
    '''
    Computes the object's instantaneous semi-major axis from its current state.

    Args:
        [state]: State vector whose elements 0:3 are the ECI position (m) and
            3:6 the ECI velocity (m s^-1).
        [config]: Simulation config, providing [mu] (m^3 s^-2).

    Returns:
        Semi-major axis (m), from the vis-viva equation.
    '''
    mag_r = np.linalg.norm(state[0:3])
    mag_v = np.linalg.norm(state[3:6])
    subtraction = (config.mu / mag_r) - ((mag_v ** 2) / 2)

    a = config.mu / (2 * subtraction)

    return a


def current_orbital_period(state, config):
    '''
    Computes the object's instantaneous orbital period from its current state.

    Args:
        [state]: State vector whose elements 0:3 are the ECI position (m) and
            3:6 the ECI velocity (m s^-1).
        [config]: Simulation config, providing [mu] (m^3 s^-2).

    Returns:
        Orbital period (s), from Kepler's third law applied to the
        instantaneous semi-major axis.
    '''
    a = current_semimajor_axis(state, config)
    period = 2 * np.pi * np.sqrt((a ** 3) / config.mu)

    return period
