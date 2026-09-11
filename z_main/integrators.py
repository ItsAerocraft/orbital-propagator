import numpy as np

from scipy.integrate import solve_ivp


def event_reentry(t, state, config):
    '''
    Computes the object's altitude above ground level, used as the re-entry
    termination event for the integrator.

    Args:
        [t]: Elapsed time since epoch (s).
        [state]: State vector whose first three elements are the ECI position (m).
        [config]: Simulation config, providing [R] (m).

    Returns:
        Altitude above ground level (m); positive above ground, negative below.
    '''
    pos_vctr = np.array(state[0:3])
    mag_r = np.linalg.norm(pos_vctr)

    return mag_r - config.R


def propagate_adaptive(function, state_0, config):
    '''
    Integrates a given dynamics function (e.g. [dynamics]) over the configured
    duration using an adaptive-step solve_ivp, terminating early on re-entry.

    Args:
        [function]: Callable with signature ([t], [state], [config]) returning
            the state derivative vector.
        [state_0]: Initial state vector, in the same units expected by [function].
        [config]: Simulation config, providing [T] (s).

    Returns:
        The scipy `OdeResult` from solve_ivp, containing up to 2000 adaptive
        time-steps between 0 and [T].

    Wraps [function] and the re-entry event to the (t, state) signature
    solve_ivp requires, since it cannot pass through [config]. The re-entry
    event only triggers on a decreasing crossing of zero (e.g. 0.1 -> 0 -> -0.1),
    not an increasing one.
    '''

    wrap_dynamics = lambda t, state: function(t, state, config)

    wrap_reentry = lambda t, state: event_reentry(t, state, config)
    wrap_reentry.terminal = True
    wrap_reentry.direction = -1

    t_eval = np.linspace(0, config.T, 2000)

    result = solve_ivp(
        fun = wrap_dynamics,
        t_span = (0, config.T),
        y0 = state_0,
        method = "RK45",
        events = wrap_reentry,
        t_eval = t_eval,
        rtol = 1e-8,  # relative tolerance (~8 significant figures) before the solver adapts its time-step
        atol = 1e-6  # absolute tolerance floor, preventing the time-step from shrinking to near-zero and crashing the solver
    )

    return result
