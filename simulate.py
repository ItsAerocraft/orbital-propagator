import time

import os

import numpy as np

from conversion.convert_elements import coe_to_cartesian, current_orbital_period

from z_main.config import Config, CONTROLLER_TYPE
from z_main.integrators import propagate_adaptive
from z_main.dynamics import dynamics

from environment.stochastic import generate_f107_trajectory

from control.synthesize import synthesize

from plotting.plot_orbit import plot_3D
from plotting.plot_altitude import plot_altitude
from plotting.plot_silhouette import plot_silhouette
from plotting.plot_stochastic import plot_stochastic


def execute():
    '''
    Runs the orbital propagation (single run or Monte Carlo stochastic batch,
    based on user input) and plots the requested output.
    '''
    start_time = time.perf_counter()
    config = Config()

    if CONTROLLER_TYPE == "contraction" and not os.path.exists('control/K.npy'):
        print("K.npy not found. Running synthesize.py...")
        synthesize()

    inertial_values_0 = coe_to_cartesian(config.a_0, config.e, config.theta_0,
                                         config.omega_0, config.inc_0, config.raan_0,
                                         config
    )
    quaternion_values_0 = config.q_0
    omega_values_0 = config.om_body_0

    state_0 = np.concatenate([inertial_values_0, quaternion_values_0, omega_values_0])

    print("")
    graph_type = input("Enter graph type or command: ")

    if graph_type in ["altitude", "orbit", "silhouette", "stochastic"]:
        print("Running simulation, graph type = "+ graph_type + "\n")

    elif graph_type == "synthesize":
        if not os.path.exists('control/K.npy'):
            print("K.npy not found. Running synthesize.py...")
            synthesize()
        else:
            print("K.npy found, nothing to change.")

    else:
        raise ValueError("Invalid graph type.")

    if graph_type == "stochastic":
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

                result = []
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

    else:
        result = []
        result = propagate_adaptive(dynamics, state_0, config)

        t_array = result.t
        state_array = result.y.T  # transpose: solve_ivp returns shape (6, steps), we need (steps, 6)

        x_array = state_array[:, 0]
        y_array = state_array[:, 1]
        z_array = state_array[:, 2]

        if result.status == 0:
            print("Status: completed, no re-entry occurrence.")
        else:
            if result.t_events[0][0] <= 86400:
                print(f"Status: completed, re-entry ooccured after {result.t_events[0][0]/86400:.2f} hour(s).\n")
            else:
                print(f"Status: completed, re-entry ooccured at day {result.t_events[0][0]/86400:.2f}.\n")

    if graph_type == "orbit":
        plot_3D(x_array, y_array, z_array)
    elif graph_type == "altitude":
        plot_altitude(t_array, state_array, config)
    elif graph_type == "silhouette":
        plot_silhouette(t_array, state_array, config)
    elif graph_type == "stochastic":
        plot_stochastic(t_array_ref, all_altitude_arrays, config)
    else:
        raise ValueError("Invalid graph type.")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")

execute()
