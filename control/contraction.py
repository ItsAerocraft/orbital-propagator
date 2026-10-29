import numpy as np

try:
    K = np.load('control/K.npy')
except FileNotFoundError:
    K = None

def contraction_control(q_e_vec, omega, tau_max):

    if K is None:
        raise FileNotFoundError(
            "control/K.npy not found! Run [python control/synthesize.py] first."
        )
    
    # Concatenate the 6-state error vector
    x_err = np.concatenate([q_e_vec, omega])
    
    # The control law: tau = K @ x_err
    # Shape check: (3x6) @ (6x1) = (3x1)
    tau = K @ x_err
    
    # Enforce physical actuator limits
    return np.clip(tau, -tau_max, tau_max)