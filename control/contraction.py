import numpy as np

K = None

def load_gain():
    '''
    Loads the saved contraction gain from control/K.npy into the module-level [K].

    Sets [K] to None if control/K.npy does not exist.
    '''
    global K

    try:
        K = np.load('control/K.npy')
    except FileNotFoundError:
        K = None

def set_gain(K_new):
    '''
    Sets the module-level contraction gain [K] directly, bypassing the saved file.

    Args:
        [K_new]: Contraction gain (3 x 6) to use.
    '''
    global K

    K = K_new

def contraction_raw(q_e_vec, omega):
    '''
    Computes the commanded torque from the contraction control law, before
    the actuator clip.

    Args:
        [q_e_vec]: 3-element error quaternion vector part (unitless).
        [omega]: 3-element body-frame angular velocity vector (rad s^-1).

    Returns:
        3-element array representing the commanded torque (N m).

    Raises:
        FileNotFoundError: If the module-level gain [K] has not been loaded
            (run control/synthesize.py first).
    '''

    if K is None:
        raise FileNotFoundError(
            "control/K.npy not found! Run [python control/synthesize.py] first."
        )

    # Concatenate the 6-state error vector
    x_err = np.concatenate([q_e_vec, omega])

    # The control law: tau = K @ x_err
    # Shape check: (3x6) @ (6x1) = (3x1)
    return K @ x_err

def contraction_control(q_e_vec, omega, tau_max):
    '''
    Computes the contraction control torque, clipped to the actuator limit.

    Args:
        [q_e_vec]: 3-element error quaternion vector part (unitless).
        [omega]: 3-element body-frame angular velocity vector (rad s^-1).
        [tau_max]: Actuator maximum torque limit per component (N m).

    Returns:
        3-element array representing the clipped commanded torque (N m).
    '''

    tau = contraction_raw(q_e_vec, omega)

    # Enforce physical actuator limits
    return np.clip(tau, -tau_max, tau_max)