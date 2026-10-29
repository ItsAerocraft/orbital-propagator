import numpy as np
import cvxpy as cp

import os

from control.jacobian import jacobian

from z_main.config import Config, LAMBDA, N_SAMPLES, OMEGA_MAX, ERROR_MAX

from tumble.inertia import inertia_cylinder, inertia_cuboid

def getI(config):
    if config.object == "cylinder":
        mass, radius, height = config.m, config.radius_cylinder, config.height_cylinder
        I = inertia_cylinder(mass, radius, height)

    elif config.object == "cuboid":
        mass, length, width, height = config.m, config.length_cuboid, config.width_cuboid, config.height_cuboid
        I = inertia_cuboid(mass, length, width, height)

    return I
     

I = getI(Config())

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

    return states

def synthesize():
    print(f"Sampling {N_SAMPLES} states...")
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

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver = cp.CLARABEL)
    
    if prob.status != cp.OPTIMAL:
        print(f"Solver failed! Status: {prob.status}")
        print("Try reducing OMEGA_MAX, ERROR_MAX, or LAMBDA.")
        return

    W_val = W.value
    Y_val = Y.value
    
    M = np.linalg.inv(W_val)
    K_scaled = Y_val @ np.linalg.inv(W_val)
    K = K_scaled @ S 
    
    os.makedirs('control', exist_ok=True)
    np.save('control/K.npy', K)
    np.save('control/M.npy', M)
    print("Solver success, saved K.npy and M.npy")

if __name__ == "__main__":
    synthesize()