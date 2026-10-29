import numpy as np

#let x represent the q_vec and omega portions of the state vector

def skew(v):
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],     0, -v[0]],
        [-v[1], v[0],     0]
    ])

def return_test_derivatives(x, I, I_inv):
    q_vec = x[0:3]
    omega = x[3:6]
    q0 = np.sqrt(1 - np.dot(q_vec, q_vec)) 

    #no control and no disturbance, so tau_c and tau_d both equal 0
    q_dot = 0.5 * (q0 * omega + np.cross(q_vec, omega))
    omega_dot = I_inv @ (- np.cross(omega, I @ omega))
    
    return np.concatenate([q_dot, omega_dot])

def jacobian(x, I, I_inv):
    q_vec = x[0:3]
    omega = x[3:6]
    q0 = np.sqrt(1 - np.dot(q_vec, q_vec))
    
    # A11: d(q_dot)/d(q)
    A11 = - 0.5 * ((1 / q0) * np.outer(omega, q_vec) + skew(omega))
    
    # A12: d(q_dot)/d(omega)
    A12 = 0.5 * (q0 * np.eye(3) + skew(q_vec))
    
    # A21: d(omega_dot)/d(q) = 0
    A21 = np.zeros((3, 3))
    
    # A22: d(omega_dot)/d(omega)
    A22 = I_inv @ ((skew(I @ omega)) - (skew(omega) @ I))
    
    A = np.block([
        [A11, A12],
        [A21, A22]
    ])
    
    return A