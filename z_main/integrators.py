import numpy as np

from scipy.integrate import solve_ivp

def event_reentry(t, state, config):
    """
    Function takes state vector and time as inputs.
    Function computes the altitude of the object as measured from ground level.
    The returned function needed to check if object has experienced collision, necessary check for solver to terminate.
    """
    pos_vctr = np.array(state[0:3]) # extract items x, y, z from state vector
    mag_r = np.linalg.norm(pos_vctr) # obtain magnitude of position vector

    return mag_r - config.R  # above ground level returns positive number, below ground level returns negative number

#wrap functions so that solve_ivp can use in form (t, state), it cannot use (t, state, config)
def propagate_adaptive(function, state_0, config):
    """
    Function takes state vector and time as inputs.
    Function computes a given function (e.g. dynamics) through solve_ivp, gives state vector array for 2000 adaptive time-steps.
    Adaptive time-steps are used to stay within relative and absolute tolerance bounds.
    """

    wrap_dynamics = lambda t, state: function(t, state, config)

    wrap_reentry = lambda t, state: event_reentry(t, state, config)
    wrap_reentry.terminal = True # stop integration when reentry_event returns 0, but usually decreasing through 0 (not integers)
    wrap_reentry.direction = -1 # event only triggers when reentry_event decreases through 0 (e.g. 0.1 --> 0 --> -0.1)

    t_eval = np.linspace(0, config.T, 2000) #for solve_ivp to save 2000 time-steps starting at 0 and ending at T

    result = solve_ivp(
        fun = wrap_dynamics,
        t_span = (0, config.T),
        y0 = state_0,
        method = "RK45",
        events = wrap_reentry, #uses reentry wrapped function, with respective termination trigger
        t_eval = t_eval, # requested 2000 time-steps
        rtol = 1e-8,  #relative tolerance, accurate to ~8 s.f. --> allowed error of solver before solver adapts time-step again to stay within bounds
        atol = 1e-6 #absolute tolerance, minimum precision threshold to prevent solve_ivp from shrinking time-step to infinitessmal size and crashing sim
    )

    return result
