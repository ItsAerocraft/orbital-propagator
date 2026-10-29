import numpy as np

from control.jacobian import jacobian
from tumble.inertia import inertia

from tumble.quaternion import quaternion_error_vec

from z_main.config import Config, Q_TARGET, LAMBDA

I = inertia(Config)
I_inv = np.linalg.inv(I)
B = np.block([[np.zeros((3, 3))], [I_inv]])

def sample_weights(t):
    """Time each sample represents (s), so fractions are of time, not of samples."""
    edges = np.concatenate([[t[0]], 0.5 * (t[1:] + t[:-1]), [t[-1]]])
    return np.diff(edges)

def audit_trajectory(x_log, M, K, t, clipped, lam = LAMBDA):
    '''
    Evaluates the contraction condition along a trajectory.

    The certificate assumes tau = K x, so clipped samples are not covered by it:
    their margins are still returned, but they are excluded from [fraction_held]
    and [worst_margin].

    Returns:
        Tuple of ([fraction_covered], [fraction_held], [worst_margin], [margins]):
        the fraction of time unclipped, the fraction of unclipped time with
        margin <= 0, the worst unclipped margin (nan if none), and the margin
        at every sample.
    '''
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
        C = A_sum.T @ M + M @ A_sum + 2 * lam * M
        C = 0.5 * (C + C.T) # to ensure  symmetry and no floating point error

        margins.append(np.linalg.eigvalsh(C).max())

    margins = np.array(margins)

    weights = sample_weights(t)
    covered = ~clipped

    fraction_covered = weights[covered].sum() / weights.sum()

    if covered.any():
        fraction_held = weights[covered & (margins <= 0)].sum() / weights[covered].sum()
        worst_margin = margins[covered].max()
    else:
        fraction_held = 0.0
        worst_margin = np.nan

    return fraction_covered, fraction_held, worst_margin, margins
