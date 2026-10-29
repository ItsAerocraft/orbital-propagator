import numpy as np
import cvxpy as cp

import os

from control.jacobian import jacobian

from z_main.config import Config, LAMBDA, N_SAMPLES, OMEGA_MAX, ERROR_MAX

from tumble.inertia import inertia


I = inertia(Config())

I_inv = np.linalg.inv(I)

B = np.block([[np.zeros((3, 3))], [I_inv]])

def sample_states(n, error_max, omega_max):
    """Generate n random states in the region."""
    states = []
    for _ in range(n):
        q = np.random.randn(3)
        q = q / np.linalg.norm(q) * np.random.uniform(0, error_max)
        
        omega = np.random.randn(3)
        omega = omega / np.linalg.norm(omega) * np.random.uniform(0, omega_max)

        states.append(np.concatenate([q, omega]))

        for q_sign in [-1, 1]:
            for omega_sign in [-1, 1]:
                q_corner = np.array([q_sign * error_max, 0, 0])
                omega_corner = np.array([omega_sign * error_max, 0, 0])

                states.append(np.concatenate([q_corner, omega_corner]))

    return states

def synthesize():
    print(f"Sampling {N_SAMPLES + 4} states...")
    states = sample_states(N_SAMPLES, ERROR_MAX, OMEGA_MAX)
    
    omega_scale = OMEGA_MAX 
    S = np.diag([1, 1, 1, 1/omega_scale, 1/omega_scale, 1/omega_scale])
    S_inv = np.diag([1, 1, 1, omega_scale, omega_scale, omega_scale])
    
    W = cp.Variable((6, 6), symmetric=True)
    Y = cp.Variable((3, 6))
    constraints = [W >> 1e-6 * np.eye(6)]
    
    print("Solving LMI...")

    for x in states:
        A = jacobian(x, I, I_inv)
        A_s = S @ A @ S_inv
        B_s = S @ B
        
        expr = A_s @ W + W @ A_s.T + B_s @ Y + Y.T @ B_s.T + 2 * LAMBDA * W
        expr = 0.5 * (expr + expr.T) 
        constraints.append(expr << 0)

    prob = cp.Problem(cp.Minimize(cp.trace(W) + cp.norm(Y, 'fro')), constraints)
    prob.solve(solver = cp.CLARABEL)
    
    if prob.status != cp.OPTIMAL:
        print(f"Solver failed! Status: {prob.status}")
        print("Try reducing OMEGA_MAX, ERROR_MAX, or LAMBDA.")
        return

    W_val = W.value
    Y_val = Y.value
    
    M_scaled = np.linalg.inv(W_val)
    M = S.T @ M_scaled @ S
    
    K_scaled = Y_val @ np.linalg.inv(W_val)
    K = K_scaled @ S 
    
    os.makedirs('control', exist_ok=True)
    np.save('control/K.npy', K)
    np.save('control/M.npy', M)
    print("Solver success, saved K.npy and M.npy")

    print(f"Max W/Y: {np.max(np.abs(W.value)):.4f}, {np.max(np.abs(Y.value)):.4f}")
    print(f"Max absolute value in K: {np.max(np.abs(K)):.2f}")

if __name__ == "__main__":
    synthesize()