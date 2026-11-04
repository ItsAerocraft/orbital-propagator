import numpy as np

def pd_raw(q_error_vec, omega, Kp, Kd):
    '''
    Computes the commanded torque from the PD control law, before the
    actuator clip.

    Args:
        [q_error_vec]: 3-element error quaternion vector part (unitless).
        [omega]: 3-element body-frame angular velocity vector (rad s^-1).
        [Kp]: Proportional gain (N m).
        [Kd]: Derivative gain (N m s).

    Returns:
        3-element array representing the commanded torque (N m).
    '''

    return -Kp * q_error_vec - Kd * omega

def pd_controller(q_error_vec, omega, Kp, Kd, tau_max):
    '''
    Computes the PD control torque, clipped to the actuator limit.

    Args:
        [q_error_vec]: 3-element error quaternion vector part (unitless).
        [omega]: 3-element body-frame angular velocity vector (rad s^-1).
        [Kp]: Proportional gain (N m).
        [Kd]: Derivative gain (N m s).
        [tau_max]: Actuator maximum torque limit per component (N m).

    Returns:
        3-element array representing the clipped commanded torque (N m).
    '''

    tau = pd_raw(q_error_vec, omega, Kp, Kd)
    return np.clip(tau, -tau_max, tau_max)
