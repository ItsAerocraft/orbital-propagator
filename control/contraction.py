import numpy as np

K = None

def load_gain():
    global K

    try:
        K = np.load('control/K.npy')
    except FileNotFoundError:
        K = None

def set_gain(K_new):
    global K

    K = K_new

def contraction_raw(q_e_vec, omega):

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

    tau = contraction_raw(q_e_vec, omega)
    
    # Enforce physical actuator limits
    return np.clip(tau, -tau_max, tau_max)