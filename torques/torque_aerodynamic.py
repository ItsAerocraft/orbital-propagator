import numpy as np

from forces.force_drag import drag_accel

from conversion.convert_body import inertial_to_body

def aerod_torque(t, state, config):
    '''
    Function takes state vector and time as inputs.
    Function computes and returns the aerodynamic torque using the state vector.
    The returned function affects angular velocity components of state vector.
    '''

    F_drag_inertial = drag_accel(t, state, config) * config.m #taking 3D force vector from 3D drag vector

    q = state[6:10]

    F_drag_body = inertial_to_body(F_drag_inertial, q) #rotating the 3D inertial force vector to the body frame

    result = np.cross(config.r_cp, F_drag_body) #standard formula for aerodynamic torque, assumed constant center of pressure position

    return result




