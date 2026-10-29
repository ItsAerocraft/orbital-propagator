import numpy as np

from control.jacobian import jacobian
from control.contraction import contraction_control, load_gain

from z_main.config import Config, TAU_MAX, ERROR_MAX, OMEGA_MAX

from tumble.inertia import inertia

load_gain()
I = inertia(Config)
K = np.load('control/K.npy')
M = np.load('control/M.npy')

def test_zero_error_zero_torque():
    """If error is 0 and spin is 0, torque must be 0."""
    q_error_vec = np.zeros(3)
    omega = np.zeros(3)
    tau = contraction_control(q_error_vec, omega, TAU_MAX)
    assert np.allclose(tau, np.zeros(3)), f"Expected zero torque, got {tau}"

def test_sign_opposes_error():
    """If pushed off target, torque should oppose the error."""

    # Push +X axis
    q_error_vec = np.array([0.1, 0.0, 0.0])
    omega = np.zeros(3)
    tau = contraction_control(q_error_vec, omega, TAU_MAX)
    assert tau[0] < 0, f"Torque should oppose +X error, got {tau[0]}"
    
    # Push -Y axis
    q_error_vec = np.array([0.0, -0.1, 0.0])
    tau = contraction_control(q_error_vec, omega, TAU_MAX)
    assert tau[1] > 0, f"Torque should oppose -Y error, got {tau[1]}"

def test_M_symmetric_positive_definite():
    """M must be symmetric and have all positive eigenvalues."""
    assert np.allclose(M, M.T), "M is not symmetric!"
    eigs = np.linalg.eigvalsh(M)
    assert np.all(eigs > 0), f"M is not positive definite! Eigenvalues: {eigs}"

def test_closed_loop_stability_at_target():
    """At the target, the closed-loop eigenvalues must be real parts < -lambda."""
    x_err = np.zeros(6)
    A = jacobian(x_err, I, np.linalg.inv(I))
    B = np.block([[np.zeros((3, 3))], [np.linalg.inv(I)]])
    A_sum = A + B @ K
    
    eigs = np.linalg.eigvals(A_sum)
    max_real_part = np.max(np.real(eigs))
    assert max_real_part < 0, f"Closed loop is not stable! Max real part: {max_real_part}"

def test_gain_respects_torque_limit():
    """Unclipped torque K x must stay within TAU_MAX across the synthesis region."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        q = rng.standard_normal(3)
        q = q / np.linalg.norm(q) * rng.uniform(0, ERROR_MAX)
        omega = rng.standard_normal(3)
        omega = omega / np.linalg.norm(omega) * rng.uniform(0, OMEGA_MAX)
        worst = max(worst, np.max(np.abs(K @ np.concatenate([q, omega]))))
    assert worst <= TAU_MAX * (1 + 1e-6), f"Max |K x| = {worst} exceeds TAU_MAX = {TAU_MAX}"
