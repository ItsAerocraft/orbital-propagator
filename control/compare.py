import os
import dataclasses

import numpy as np

from z_main.config import TAU_MAX, Q_TARGET, COMPARE_DURATION, SETTLE_BAND
from z_main.integrators import propagate_adaptive
from z_main.dynamics import dynamics

from torques.torque_controller import get_control_torque, get_raw_control_torque, pd_gains
from torques.torque_gravitygradient import gravgrad_torque
from torques.torque_aerodynamic import aerod_torque

from tumble.quaternion import quaternion_error_vec

from control.synthesize import synthesize, synth_params
from control.contraction import load_gain
from control.audit import audit_trajectory, sample_weights


CONTROLLERS = ["pd", "contraction"]

TRANSIENT_WINDOW = 20 # (s) [initial window sampled densely to resolve the transient]
TRANSIENT_SAMPLES = 2001 # (unitless) [samples over TRANSIENT_WINDOW; 0.01 s spacing]


def sample_times(T):
    '''
    Output times for the comparison: 0.01 s spacing over the first
    TRANSIENT_WINDOW seconds, then the usual 2000 samples over the rest.
    '''
    dense = np.linspace(0, TRANSIENT_WINDOW, TRANSIENT_SAMPLES)
    coarse = np.linspace(TRANSIENT_WINDOW, T, 2000)

    return np.unique(np.concatenate([dense, coarse]))


def attitude_gain(cfg, K):
    '''
    Returns the 3x3 attitude block K_q of the control law tau = K_q q_e + K_w omega.
    '''
    if cfg.controller == "pd":
        return -pd_gains(cfg)[0] * np.eye(3)

    return K[:, :3]


def run_controller(name, config, state_0, K):
    '''
    Propagates the system over the comparison window with one controller and
    computes its performance metrics.

    Args:
        [name]: Controller type (pd / contraction).
        [config]: Simulation config, copied with [controller] = [name] and
            [T] = COMPARE_DURATION.
        [state_0]: Initial 13-element state vector.
        [K]: Contraction gain (3 x 6), used for the steady-state prediction.

    Returns:
        Dict with [t] (s), [states] (N x 13), [error_deg] (deg), [torque]
        (N x 3, N m), [clipped] (N bools), and the metrics [settling_time]
        (s, or None if it did not settle), [steady_state_error] (deg),
        [predicted_ss_error] (deg), [peak_torque] (N m), [effort]
        (N^2 m^2 s), [clip_count] (samples), [tail_control_torque] and
        [tail_disturbance_torque] (N m).

    The torque history is rebuilt from the sampled states by calling the same
    controller function the dynamics used, so it is exact at each sample.
    Time-averaged metrics are weighted by sample spacing, since the output is
    denser over the transient.
    '''
    cfg = dataclasses.replace(config, controller = name, T = COMPARE_DURATION)

    result = propagate_adaptive(dynamics, state_0, cfg, sample_times(cfg.T))

    t_array = result.t
    state_array = result.y.T

    error_vecs = np.array([quaternion_error_vec(state[6:10], Q_TARGET) for state in state_array])
    error_deg = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(error_vecs, axis = 1), 0, 1)))

    torque = np.array([get_control_torque(state[6:10], state[10:13], cfg) for state in state_array])
    raw_torque = np.array([get_raw_control_torque(state[6:10], state[10:13], cfg) for state in state_array])

    clipped = np.any(np.abs(raw_torque) > TAU_MAX, axis = 1)

    # settled after the last sample with error above the band
    outside = np.where(error_deg > SETTLE_BAND)[0]

    if len(outside) == 0:
        settling_time = 0.0
    elif outside[-1] == len(t_array) - 1:
        settling_time = None
    else:
        settling_time = t_array[outside[-1] + 1]

    # steady state is the last 10 % of the window, by time
    tail = t_array >= 0.9 * cfg.T
    tail_weights = sample_weights(t_array)[tail]

    tail_mean = lambda values: np.average(values, weights = tail_weights)

    # at equilibrium tau_c = K_q q_e cancels the disturbance, so q_e = K_q^-1 tau_c and theta = 2 arcsin |q_e|
    K_q_inv = np.linalg.inv(attitude_gain(cfg, K))
    q_pred = (K_q_inv @ torque[tail].T).T
    predicted_deg = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(q_pred, axis = 1), 0, 1)))

    disturbance = np.array([gravgrad_torque(t, state, cfg) + aerod_torque(t, state, cfg)
                            for t, state in zip(t_array[tail], state_array[tail])])

    return {
        "t": t_array,
        "states": state_array,
        "error_deg": error_deg,
        "torque": torque,
        "clipped": clipped,
        "settling_time": settling_time,
        "steady_state_error": tail_mean(error_deg[tail]),
        "predicted_ss_error": tail_mean(predicted_deg),
        "peak_torque": np.abs(torque).max(),
        "effort": np.trapezoid(np.sum(torque ** 2, axis = 1), t_array),
        "clip_count": int(clipped.sum()),
        "tail_control_torque": tail_mean(np.linalg.norm(torque[tail], axis = 1)),
        "tail_disturbance_torque": tail_mean(np.linalg.norm(disturbance, axis = 1)),
    }


def print_table(results):
    '''
    Prints the controller comparison metrics as a table, one column per controller.

    Args:
        [results]: Dict mapping controller name to its [run_controller] result.
    '''
    names = list(results)

    rows = [
        ("Settling time / s", lambda r: "did not settle" if r["settling_time"] is None else f"{r['settling_time']:.2f}"),
        ("Steady-state error / deg", lambda r: f"{r['steady_state_error']:.4g}"),
        ("Predicted error (2 tau / K_p) / deg", lambda r: f"{r['predicted_ss_error']:.4g}"),
        ("Measured / predicted error", lambda r: f"{r['steady_state_error'] / r['predicted_ss_error']:.3f}"),
        ("Steady-state |tau_c| / N m", lambda r: f"{r['tail_control_torque']:.4g}"),
        ("Steady-state |tau_dist| / N m", lambda r: f"{r['tail_disturbance_torque']:.4g}"),
        ("Peak torque / N m", lambda r: f"{r['peak_torque']:.4g}"),
        ("Effort int |tau|^2 dt / N^2 m^2 s", lambda r: f"{r['effort']:.4g}"),
        ("Torque clips / samples", lambda r: f"{r['clip_count']} of {len(r['t'])}"),
    ]

    label_width = max(len(label) for label, _ in rows)
    col_width = max(16, *(len(name) for name in names))

    header = f"{'Metric':<{label_width}}" + "".join(f" | {name:>{col_width}}" for name in names)

    print(header)
    print("-" * len(header))

    for label, fmt in rows:
        print(f"{label:<{label_width}}" + "".join(f" | {fmt(results[name]):>{col_width}}" for name in names))


def gain_is_current():
    '''
    Checks the saved gain was synthesised with the current config parameters.
    '''
    paths = ['control/K.npy', 'control/M.npy', 'control/synth_params.npy', 'control/lambda.npy']

    if not all(os.path.exists(path) for path in paths):
        return False

    saved = np.load('control/synth_params.npy')

    return saved.shape == synth_params().shape and np.allclose(saved, synth_params())


def compare_controllers(config, state_0):
    '''
    Runs every controller from the same initial state, then prints the
    comparison table and contraction audit.

    Args:
        [config]: Simulation config.
        [state_0]: Initial 13-element state vector.

    Returns:
        Tuple of ([results], [margins]): the dict mapping controller name to
        its [run_controller] result, and the contraction audit margin at each
        sample, for passing to [plotting.plot_control.plot_comparison].
    '''
    if not gain_is_current():
        print("Saved gain missing or stale. Running synthesize.py...")
        x0_err = np.concatenate([quaternion_error_vec(state_0[6:10], Q_TARGET), state_0[10:13]])
        synthesize(x0_err)

    load_gain()

    K = np.load('control/K.npy')
    M = np.load('control/M.npy')
    lam = float(np.load('control/lambda.npy'))

    results = {}

    for name in CONTROLLERS:
        print(f"Running {name} controller for {COMPARE_DURATION} s...")
        results[name] = run_controller(name, config, state_0, K)

    print(f"\nController comparison ({COMPARE_DURATION} s, settled below {SETTLE_BAND} deg)\n")
    print_table(results)

    run = results["contraction"]

    fraction_covered, fraction_held, worst_margin, margins = audit_trajectory(run["states"], M, K, run["t"], run["clipped"], lam)

    verdict = "PASS" if worst_margin <= 0 else "FAIL"

    print(f"\nContraction audit (lambda = {lam:.3g}): {verdict}, covered {fraction_covered:.1%} of time (unclipped), "
          f"condition held for {fraction_held:.1%} of covered time, worst margin {worst_margin:.4g}\n")

    return results, margins
