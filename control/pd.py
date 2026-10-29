import numpy as np

def pd_raw(q_error_vec, omega, Kp, Kd):

    return -Kp * q_error_vec - Kd * omega

def pd_controller(q_error_vec, omega, Kp, Kd, tau_max):

    tau = pd_raw(q_error_vec, omega, Kp, Kd)
    return np.clip(tau, -tau_max, tau_max)
