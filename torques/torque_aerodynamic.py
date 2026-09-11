import numpy as np

from forces.force_drag import drag_accel

from conversion.convert_body import inertial_to_body


def aerod_torque(t, state, config):
    '''
    Computes the aerodynamic torque acting on the object.

    Args:
        [t]: Elapsed time since epoch (s).
        [state]: State vector whose elements 6:10 are the orientation
            quaternion (unitless).
        [config]: Simulation config, providing [m] (mass, kg) and [r_cp]
            (vector from center of mass to center of pressure, m).

    Returns:
        3-element array representing the aerodynamic torque (N m).

    Assumes a constant center of pressure position. The inertial drag force
    is rotated into the body frame before being crossed with [r_cp].
    '''

    F_drag_inertial = drag_accel(t, state, config) * config.m

    q = state[6:10]

    F_drag_body = inertial_to_body(F_drag_inertial, q)

    result = np.cross(config.r_cp, F_drag_body)

    return result
