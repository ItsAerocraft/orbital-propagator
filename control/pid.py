import numpy as np

class PIDController:
    def __init__(self, K_p, K_d, K_i, tau_max, dt = 0.01):
        self.K_p = K_p
        self.K_d = K_d
        self.K_i = K_i
        self.tau_max = tau_max
        self.dt = dt
        self.integral_error = np.zeros(3)

    def reset(self):
        self.integral_error = np.zeros(3)

    def update(self, q_error_vec, omega):
        self.integral_error += q_error_vec * self.dt
        
        tau = -self.K_p * q_error_vec - self.K_d * omega - self.K_i * self.integral_error
        
        return np.clip(tau, -self.tau_max, self.tau_max)