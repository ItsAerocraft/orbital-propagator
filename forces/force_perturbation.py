import numpy as np

def J2_accel(t, state, config):
    """
    Function returns acceleration due to J2 perturbation using standard formulae.
    """

    pos_vctr = np.array(state[0:3])
    x, y, z = pos_vctr
    mag_r = np.linalg.norm(pos_vctr)
    
    J2_factor = - (3 * config.J2 * config.mu * (config.R ** 2)) / (2 * (mag_r ** 5))
    J2_const1 = (5 * (z ** 2) / (mag_r ** 2))

    return np.array([J2_factor * (1 - J2_const1) * x,
                     J2_factor * (1 - J2_const1) * y,
                     J2_factor * (3 - J2_const1) * z
    ])

def J3_accel(t, state,config):
    """
    Function returns acceleration due to J3 perturbation using standard formulae.
    """

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
    """
    Function returns acceleration due to J4 perturbation using standard formulae.
    """

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
    """
    Function sums all 3D acceleration vectors due to J2, J3, and J4 perturbation into one 3D acceleration vector.
    """
    return J2_accel(t, state, config) + J3_accel(t, state, config) + J4_accel(t, state, config)
