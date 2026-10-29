import numpy as np

from conversion.convert_body import inertial_to_body

from tumble.inertia import inertia


def gravgrad_torque(t, state, config):
    '''
    Computes the gravity-gradient torque acting on the object.

    Args:
        [t]: Elapsed time since epoch (s), unused (torque depends only on
            position and orientation).
        [state]: State vector whose elements 0:3 are the ECI position (m)
            and 6:10 the orientation quaternion (unitless).
        [config]: Simulation config, providing [object] (shape flag), [m]
            (mass, kg), [mu] (m^3 s^-2), and the relevant dimensions:
            [radius_cylinder] (m), [height_cylinder] (m) for a cylinder, or
            [length_cuboid] (m), [width_cuboid] (m), [height_cuboid] (m) for
            a cuboid.

    Returns:
        3-element array representing the gravity-gradient torque (N m).

    The position vector is rotated into the body frame before being combined
    with the object's moment of inertia matrix, per the standard
    gravity-gradient torque formula.
    '''

    x, y, z = state[0:3]
    q = state[6:10]
    mag_r = np.linalg.norm(state[0:3])

    r_hat_inertial = state[0:3] / mag_r
    r_hat_body = inertial_to_body(r_hat_inertial, q)

    I = inertia(config)

    I_mult_r_hat = I @ r_hat_body

    result = ((3 * config.mu) / (mag_r ** 3)) * np.cross(r_hat_body, I_mult_r_hat)

    return result
