import dataclasses

import numpy as np

from z_main.config import Q_TARGET

from tumble.inertia import inertia
from tumble.quaternion import quaternion_error_vec

from control.synthesize import synthesize
from control.contraction import set_gain, load_gain
from control.compare import run_controller


PD_OMEGA_N = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5] # (rad s^-1) [PD natural frequencies swept, zeta = 1]
PD_ZETA = 1.0 # (unitless) [PD damping ratio for the sweep]

CONTRACTION_LAMBDAS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6] # (s^-1) [contraction rates swept, each re-synthesised]

SYNTH_SEED = 0 # (integer) [seed for the LMI state samples, so every rate sees the same samples]


def pd_gains_for(omega_n, zeta, I_axis):
    '''
    Returns PD gains ([Kp], [Kd]) placing the small-angle poles at [omega_n]
    with damping [zeta]. Kp carries a factor 2 because q_e = sin(theta / 2).
    '''
    return 2 * I_axis * omega_n ** 2, 2 * zeta * omega_n * I_axis


def summarise(run, **params):
    '''
    Keeps the scalar metrics of a [run_controller] result.
    '''
    keys = ["settling_time", "steady_state_error", "predicted_ss_error", "effort", "peak_torque", "clip_count"]

    return {**params, **{key: run[key] for key in keys}}


def run_sweep(config, state_0):
    '''
    Runs PD over PD_OMEGA_N and contraction over CONTRACTION_LAMBDAS from the
    same initial state, re-synthesising the contraction gain at every rate.

    Returns:
        Dict with [pd] and [contraction] lists of metric dicts, one per gain,
        plus [infeasible], the rates the LMI could not solve.
    '''
    I_axis = np.max(np.diag(inertia(config)))

    pd_points = []

    for omega_n in PD_OMEGA_N:
        kp, kd = pd_gains_for(omega_n, PD_ZETA, I_axis)
        print(f"PD: omega_n = {omega_n} rad/s (Kp = {kp:.3g}, Kd = {kd:.3g})")

        cfg = dataclasses.replace(config, kp = kp, kd = kd)
        run = run_controller("pd", cfg, state_0, None)

        pd_points.append(summarise(run, omega_n = omega_n))

    x0_err = np.concatenate([quaternion_error_vec(state_0[6:10], Q_TARGET), state_0[10:13]])

    contraction_points = []
    infeasible = []

    for lam in CONTRACTION_LAMBDAS:
        print(f"Contraction: lambda = {lam} s^-1")

        np.random.seed(SYNTH_SEED)
        synth = synthesize(x0_err, lambdas = [lam], save = False)

        if synth is None:
            infeasible.append(lam)
            continue

        K = synth[0]
        set_gain(K)

        run = run_controller("contraction", config, state_0, K)

        contraction_points.append(summarise(run, lam = lam))

    # restore the saved gain for anything run afterwards
    load_gain()

    return {"pd": pd_points, "contraction": contraction_points, "infeasible": infeasible}


def print_sweep(sweep):
    '''
    Prints one row per swept gain.
    '''
    fmt_settle = lambda t: "did not settle" if t is None else f"{t:.2f}"

    print(f"\n{'Controller':<24} | {'Effort / N^2 m^2 s':>18} | {'Offset / deg':>12} | {'Settling / s':>14} | {'Clips':>6}")
    print("-" * 86)

    for p in sweep["pd"]:
        print(f"{'pd, wn = ' + str(p['omega_n']):<24} | {p['effort']:>18.4g} | {p['steady_state_error']:>12.4g} | {fmt_settle(p['settling_time']):>14} | {p['clip_count']:>6}")

    for p in sweep["contraction"]:
        print(f"{'contraction, lam = ' + str(p['lam']):<24} | {p['effort']:>18.4g} | {p['steady_state_error']:>12.4g} | {fmt_settle(p['settling_time']):>14} | {p['clip_count']:>6}")

    if sweep["infeasible"]:
        print(f"\nInfeasible contraction rates (no gain): {sweep['infeasible']}")
