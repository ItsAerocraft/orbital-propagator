import numpy as np

from control.jacobian import jacobian
from tumble.inertia import inertia

from tumble.quaternion import quaternion_error_vec

from z_main.config import Config, Q_TARGET, LAMBDA

I = inertia(Config)
I_inv = np.linalg.inv(I)
B = np.block([[np.zeros((3, 3))], [I_inv]])

def audit_trajectory(x_log, M, K):
    margins = []
    
    for row in x_log:
        # Extract state and compute error
        q_current = row[6:10]
        omega = row[10:13]
        q_e_vec = quaternion_error_vec(q_current, Q_TARGET)
        x_err = np.concatenate([q_e_vec, omega])
        
        A = jacobian(x_err, I, I_inv)
        A_sum = A + B @ K
        
        # Compute C(x)
        C = A_sum.T @ M + M @ A_sum + 2 * LAMBDA * M
        C = 0.5 * (C + C.T) # to ensure  symmetry and no floating point error
        
        margins.append(np.linalg.eigvalsh(C).max())
        
    margins = np.array(margins)
    fraction_held = (margins <= 0).mean()
    worst_margin = margins.max()
    
    return fraction_held, worst_margin, margins