import numpy as np

def pd_controller(q_error_vec, omega, Kp, Kd, tau_max):

    tau = -Kp * q_error_vec - Kd * omega
    return np.clip(tau, -tau_max, tau_max)