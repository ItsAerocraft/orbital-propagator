import time

import numpy as np

from conversion.convert_elements import coe_to_cartesian, current_orbital_period

from z_main.config import Config
from z_main.integrators import propagate_adaptive
from z_main.dynamics import dynamics

from environment.stochastic import generate_f107_trajectory

from control.compare import compare_controllers

from plotting.plot_single import plot_single_run
from plotting.plot_stochastic import plot_stochastic
from plotting.plot_control import plot_comparison


MODES = ["single", "stochastic", "control"]


def initial_state(config):
    '''
    Builds the initial 13-element state vector from the config.

    Args:
        [config]: Simulation config, providing the classical orbital elements,
            [q_0] and [om_body_0].

    Returns:
        13-element array of ECI position (m), ECI velocity (m s^-1),
        orientation quaternion (unitless) and body-frame angular velocity (rad s^-1).
    '''
    inertial_values_0 = coe_to_cartesian(config.a_0, config.e, config.theta_0,
                                         config.omega_0, config.inc_0, config.raan_0,
                                         config
    )
    quaternion_values_0 = config.q_0
    omega_values_0 = config.om_body_0

    return np.concatenate([inertial_values_0, quaternion_values_0, omega_values_0])


def run_single(config, state_0):
    '''
    Runs a single propagation over the configured duration and prints its
    re-entry status.

    Args:
        [config]: Simulation config.
        [state_0]: Initial 13-element state vector.

    Returns:
        Tuple of ([t_array], [state_array]): sample times (s) and state
        vectors, one per row.
    '''
    result = propagate_adaptive(dynamics, state_0, config)

    t_array = result.t
    state_array = result.y.T  # transpose: solve_ivp returns shape (13, steps), we need (steps, 13)

    if result.status == 0:
        print("Status: completed, no re-entry occurrence.")
    else:
        if result.t_events[0][0] <= 86400:
            print(f"Status: completed, re-entry ooccured after {result.t_events[0][0]/86400:.2f} hour(s).\n")
        else:
            print(f"Status: completed, re-entry ooccured at day {result.t_events[0][0]/86400:.2f}.\n")

    return t_array, state_array


def run_stochastic(config, state_0):
    '''
    Runs the Monte Carlo batch, each run with its own stochastic F10.7
    trajectory, and prints how many runs re-entered.

    Args:
        [config]: Simulation config, providing [runs] and [T] (s).
        [state_0]: Initial 13-element state vector.

    Returns:
        Tuple of ([t_array_ref], [all_altitude_arrays]): the reference sample
        times (s) and each run's perigee altitude (m) interpolated onto them.
    '''
    config.stochastic = True
    runs = config.runs

    reentry_occurrence_count = 0

    all_altitude_arrays = []
    reentry_array = []

    t_array_ref = np.linspace(0, config.T, 500)

    for run_index in range(runs):
        try:
            t_samples, f107_samples = generate_f107_trajectory(config, seed = run_index)
            config.f107_trajectory = (t_samples, f107_samples)

            result = propagate_adaptive(dynamics, state_0, config)

            t_array = result.t
            state_array = result.y.T

            altitude_array = np.linalg.norm(state_array[ : , 0:3], axis = 1) - config.R

            t_array_peri = []
            altitude_array_peri = []

            for index in range(1, len(altitude_array) - 1):
                value = altitude_array[index]
                previous_value = altitude_array[index - 1]
                next_value = altitude_array[index + 1]

                local_period = current_orbital_period(state_array[index], config)

                if abs(config.a_0 - (value + config.R)) < 0.05 * (config.a_0 * (1 - config.e)):

                    if value < previous_value and value < next_value:

                        if len(t_array_peri) == 0:
                            t_array_peri.append(t_array[index])
                            altitude_array_peri.append(altitude_array[index])

                        elif len(t_array_peri) > 0:
                            previous_accepted_time = t_array_peri[-1]

                            if t_array[index] - previous_accepted_time > (0.8 * local_period):
                                t_array_peri.append(t_array[index])
                                altitude_array_peri.append(altitude_array[index])

            altitude_array_interp = np.interp(t_array_ref, t_array_peri, altitude_array_peri, left = np.nan, right = np.nan)

            all_altitude_arrays.append(altitude_array_interp)

            if result.status == 1:
                reentry_occurrence_count += 1
                reentry_array.append(run_index)

        except Exception as e:
            print(f"{run_index} failed: {e}")
            continue

    print("Status: completed, re-entry occured", reentry_occurrence_count, "time(s).")

    if reentry_occurrence_count != 0:
        print("re-entry occured on runs", reentry_array, ".")

    return t_array_ref, all_altitude_arrays


def execute():
    '''
    Asks for a mode (single / stochastic / control), runs it, and plots its graphs.
    '''
    print("")
    print("Please visit config.py (in folder z_main), to modify constants.")
    mode = input(f"Enter mode ({' / '.join(MODES)}): ").strip().lower()

    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode!r} (expected one of {', '.join(MODES)}).")

    print("Running simulation, mode = " + mode + "\n")

    start_time = time.perf_counter()
    config = Config()

    # single and stochastic study uncontrolled tumbling; control mode sets its own controllers
    if mode in ["single", "stochastic"]:
        config.controller = "none"

    state_0 = initial_state(config)

    if mode == "single":
        t_array, state_array = run_single(config, state_0)

    elif mode == "stochastic":
        t_array_ref, all_altitude_arrays = run_stochastic(config, state_0)

    elif mode == "control":
        results, margins = compare_controllers(config, state_0)

    elapsed_time = time.perf_counter() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")

    if mode == "single":
        plot_single_run(t_array, state_array, config)

    elif mode == "stochastic":
        plot_stochastic(t_array_ref, all_altitude_arrays, config)

    elif mode == "control":
        plot_comparison(results, margins)


if __name__ == "__main__":
    execute()
