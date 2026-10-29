import os
import dataclasses

import numpy as np

from z_main.config import TAU_MAX, Q_TARGET, COMPARE_DURATION, SETTLE_BAND
from z_main.integrators import propagate_adaptive
from z_main.dynamics import dynamics

from torques.torque_controller import get_control_torque

from tumble.quaternion import quaternion_error_vec

from control.synthesize import synthesize
from control.contraction import load_gain
from control.audit import audit_trajectory


CONTROLLERS = ["pd", "contraction"]


def run_controller(name, config, state_0):
    '''
    Propagates the system over the comparison window with one controller and
    computes its performance metrics.

    Args:
        [name]: Controller type (pd / contraction).
        [config]: Simulation config, copied with [controller] = [name] and
            [T] = COMPARE_DURATION.
        [state_0]: Initial 13-element state vector.

    Returns:
        Dict with [t] (s), [states] (N x 13), [error_deg] (deg), [torque]
        (N x 3, N m), and the metrics [settling_time] (s, or None if it did
        not settle), [steady_state_error] (deg), [peak_torque] (N m) and
        [clip_count] (samples).

    The torque history is rebuilt from the sampled states by calling the same
    controller function the dynamics used, so it is exact at each sample.
    '''
    cfg = dataclasses.replace(config, controller = name, T = COMPARE_DURATION)

    result = propagate_adaptive(dynamics, state_0, cfg)

    t_array = result.t
    state_array = result.y.T

    error_vecs = np.array([quaternion_error_vec(state[6:10], Q_TARGET) for state in state_array])
    error_deg = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(error_vecs, axis = 1), 0, 1)))

    torque = np.array([get_control_torque(state[6:10], state[10:13], cfg) for state in state_array])

    # settled after the last sample with error above the band
    outside = np.where(error_deg > SETTLE_BAND)[0]

    if len(outside) == 0:
        settling_time = 0.0
    elif outside[-1] == len(t_array) - 1:
        settling_time = None
    else:
        settling_time = t_array[outside[-1] + 1]

    tail = error_deg[int(0.9 * len(error_deg)) : ]

    return {
        "t": t_array,
        "states": state_array,
        "error_deg": error_deg,
        "torque": torque,
        "settling_time": settling_time,
        "steady_state_error": tail.mean(),
        "peak_torque": np.abs(torque).max(),
        "clip_count": int(np.any(np.isclose(np.abs(torque), TAU_MAX), axis = 1).sum()),
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
        ("Peak torque / N m", lambda r: f"{r['peak_torque']:.4g}"),
        ("Torque clips / samples", lambda r: f"{r['clip_count']} of {len(r['t'])}"),
    ]

    label_width = max(len(label) for label, _ in rows)
    col_width = max(16, *(len(name) for name in names))

    header = f"{'Metric':<{label_width}}" + "".join(f" | {name:>{col_width}}" for name in names)

    print(header)
    print("-" * len(header))

    for label, fmt in rows:
        print(f"{label:<{label_width}}" + "".join(f" | {fmt(results[name]):>{col_width}}" for name in names))


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
    if not os.path.exists('control/K.npy'):
        print("K.npy not found. Running synthesize.py...")
        synthesize()

    load_gain()

    results = {}

    for name in CONTROLLERS:
        print(f"Running {name} controller for {COMPARE_DURATION} s...")
        results[name] = run_controller(name, config, state_0)

    print(f"\nController comparison ({COMPARE_DURATION} s, settled below {SETTLE_BAND} deg)\n")
    print_table(results)

    K = np.load('control/K.npy')
    M = np.load('control/M.npy')

    fraction_held, worst_margin, margins = audit_trajectory(results["contraction"]["states"], M, K)

    verdict = "PASS" if worst_margin <= 0 else "FAIL"

    print(f"\nContraction audit: {verdict}, condition held at {fraction_held:.1%} of samples, worst margin {worst_margin:.4g}\n")

    return results, margins
