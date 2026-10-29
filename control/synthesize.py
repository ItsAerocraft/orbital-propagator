import numpy as np
import cvxpy as cp

import os

from control.jacobian import jacobian

from z_main.config import Config, LAMBDA, N_SAMPLES, OMEGA_MAX, ERROR_MAX, TAU_MAX

from tumble.inertia import inertia


I = inertia(Config())

I_inv = np.linalg.inv(I)

B = np.block([[np.zeros((3, 3))], [I_inv]])

LAMBDA_FALLBACKS = [LAMBDA, LAMBDA / 2, LAMBDA / 4]

def synth_params():
    """Parameters the saved K and M depend on, used to detect a stale gain."""
    return np.array([LAMBDA, N_SAMPLES, OMEGA_MAX, ERROR_MAX, TAU_MAX])

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
            omega_corner = np.array([omega_sign * omega_max, 0, 0])

            states.append(np.concatenate([q_corner, omega_corner]))

    return states

def contained_points(x0_err):
    """Extra points the invariant ellipsoid must contain, beyond the sampled region."""
    if x0_err is None:
        return []

    return [np.asarray(x0_err)]

def solve_lmi(states, points, lam, S, S_inv):
    W = cp.Variable((6, 6), symmetric=True)
    Y = cp.Variable((3, 6))
    constraints = [W >> 1e-6 * np.eye(6)]

    for x in states:
        A = jacobian(x, I, I_inv)
        A_s = S @ A @ S_inv
        B_s = S @ B

        expr = A_s @ W + W @ A_s.T + B_s @ Y + Y.T @ B_s.T + 2 * lam * W
        expr = 0.5 * (expr + expr.T)
        constraints.append(expr << 0)

    # |tau_i| <= TAU_MAX everywhere inside the ellipsoid x_s^T W^-1 x_s <= 1
    for i in range(3):
        row = Y[i:i + 1, :]
        constraints.append(cp.bmat([[np.array([[TAU_MAX ** 2]]), row], [row.T, W]]) >> 0)

    # the ellipsoid must contain the operating region, otherwise the torque bound is vacuous:
    # |q| <= ERROR_MAX and |omega_s| <= 1 imply |q|^2 / (2 ERROR_MAX^2) + |omega_s|^2 / 2 <= 1
    constraints.append(W >> np.diag([2 * ERROR_MAX ** 2] * 3 + [2] * 3))

    for p in points:
        p_s = (S @ p).reshape(6, 1)
        constraints.append(cp.bmat([[np.ones((1, 1)), p_s.T], [p_s, W]]) >> 0)

    prob = cp.Problem(cp.Minimize(cp.trace(W) + cp.norm(Y, 'fro')), constraints)
    prob.solve(solver = cp.CLARABEL)

    return prob, W, Y

def synthesize(x0_err=None, lambdas=None, save=True):
    '''
    Solves the contraction LMI, trying each rate in [lambdas] (default
    LAMBDA_FALLBACKS) until one is feasible.

    Returns:
        Tuple of ([K], [M], [lam]), or None if every rate was infeasible.
        With [save] = True the result is also written to control/.
    '''
    if lambdas is None:
        lambdas = LAMBDA_FALLBACKS

    print(f"Sampling {N_SAMPLES + 4} states...")
    states = sample_states(N_SAMPLES, ERROR_MAX, OMEGA_MAX)
    points = contained_points(x0_err)

    omega_scale = OMEGA_MAX
    S = np.diag([1, 1, 1, 1/omega_scale, 1/omega_scale, 1/omega_scale])
    S_inv = np.diag([1, 1, 1, omega_scale, omega_scale, omega_scale])

    for lam in lambdas:
        print(f"Solving LMI with lambda = {lam:.3g}...")
        prob, W, Y = solve_lmi(states, points, lam, S, S_inv)

        if prob.status == cp.OPTIMAL:
            break

        print(f"Solver failed! Status: {prob.status}")
    else:
        print("Try reducing OMEGA_MAX, ERROR_MAX, or LAMBDA.")
        return

    W_val = W.value
    Y_val = Y.value

    M_scaled = np.linalg.inv(W_val)
    M = S.T @ M_scaled @ S

    K_scaled = Y_val @ np.linalg.inv(W_val)
    K = K_scaled @ S

    if save:
        os.makedirs('control', exist_ok=True)
        np.save('control/K.npy', K)
        np.save('control/M.npy', M)
        np.save('control/synth_params.npy', synth_params())
        np.save('control/lambda.npy', np.array(lam))
        print(f"Solver success with lambda = {lam:.3g}, saved K.npy and M.npy")
    else:
        print(f"Solver success with lambda = {lam:.3g}")

    print(f"Max W/Y: {np.max(np.abs(W.value)):.4f}, {np.max(np.abs(Y.value)):.4f}")
    print(f"Max absolute value in K: {np.max(np.abs(K)):.4g}")
    print(f"Max |K x| over sampled states: {max(np.max(np.abs(K @ x)) for x in states):.4g} N m (limit {TAU_MAX} N m)")

    return K, M, lam

if __name__ == "__main__":
    synthesize()
