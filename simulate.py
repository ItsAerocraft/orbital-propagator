import os
import time

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from conversion.convert_elements import coe_to_cartesian, current_orbital_period

from tumble.area import projected_area_cylinder, projected_area_cuboid

from z_main.config import Config
from z_main.integrators import propagate_adaptive
from z_main.dynamics import dynamics

from environment.stochastic import generate_f107_trajectory


def minorgridlines():
    plt.minorticks_on()
    plt.grid(True, which = "major", alpha = 0.3, linestyle = "--", linewidth = 1.5)
    plt.grid(True, which = "minor", alpha = 0.15, linestyle = ":", linewidth = 0.8)

def plot_3D(x_array, y_array, z_array):
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111, projection = "3d")

    x_range = x_array.max() - x_array.min() + 1
    y_range = y_array.max() - y_array.min() + 1
    z_range = z_array.max() - z_array.min() + 1
    
    ax.set_box_aspect([x_range, y_range, z_range])

    ax.plot(x_array, y_array, z_array)

    ax.scatter([0], [0], [0], color = "red", s = 50, label = "Earth center")
    ax.legend()

    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")

    plt.show()

def plot_altitude(t_array, state_array, config):
    altitude_array = np.linalg.norm(state_array[ : , 0:3], axis = 1) - config.R
    # [ : , 0:3] is array shape above, axis = 1 dictates that norm calculated at all rows independently

    plt.figure(figsize = (10, 6))
    plt.plot(t_array / 86400, altitude_array / 1000)

    minorgridlines()

    plt.xlabel("Time / day", fontsize = 12)
    plt.ylabel("Altitude / km", fontsize = 12)
    plt.title("Altitude against Time", fontsize = 14)

    plt.show()

def plot_omega(t_array, state_array, config):
    om_x_array, om_y_array, om_z_array = state_array[ : , 10], state_array[ : , 11], state_array[ : , 12]
    angvel_array = np.linalg.norm(state_array[ : , 10:13], axis = 1)

    plt.figure(figsize = (10, 6))
    plt.plot(t_array / 86400, om_x_array, label = "x component")
    plt.plot(t_array / 86400, om_y_array, label = "y component")
    plt.plot(t_array / 86400, om_z_array, label = "z component")
    plt.plot(t_array / 86400, angvel_array, label = "magnitude")
    
    minorgridlines()

    plt.xlabel("Time / day", fontsize = 12)
    plt.ylabel("Angular Velocity / rad$^{-1}$", fontsize = 12)
    plt.title("Angular Velocity against Time", fontsize = 14)

    plt.legend()
    plt.show()

def plot_silhouette(t_array, state_array, config):
    area_array = np.array([
        projected_area_cuboid(t, state_array[i], config) if config.object == "cuboid"
        else projected_area_cylinder(t, state_array[i], config)
        for i, t in enumerate(t_array) 
        # creates two lists, one for i, and its corresponding ith position in t_array
        #then loops through t values and i values (so state_array gets i'th position) and calls functions repeatedly
    ])

    plt.figure(figsize = (10,6))
    plt.plot(t_array / 86400, area_array)

    minorgridlines()

    plt.xlabel("Time / day", fontsize = 12)
    plt.ylabel("Silhouette Area / m$^2$", fontsize = 12)
    plt.title("Silhouette Area against Time")

    plt.show()

def plot_stochastic(t_array_ref, all_altitude_arrays, config):
    plt.figure(figsize = (10, 6))

    min_alt = np.nanmin(all_altitude_arrays, axis = 0)
    max_alt = np.nanmax(all_altitude_arrays, axis = 0)
    avg_alt = np.nanmean(all_altitude_arrays, axis = 0)

    plt.plot(t_array_ref / 86400, avg_alt / 1000, label = "average")
    plt.scatter(t_array_ref / 86400, min_alt / 1000, s = 10, color = "red", label = "minimum")
    plt.scatter(t_array_ref / 86400, max_alt / 1000, s = 10, color = "blue", label = "maximum")

    plt.fill_between(t_array_ref / 86400, max_alt / 1000, min_alt / 1000, color = "grey", alpha = 0.4)

    minorgridlines()

    
    plt.xlabel("Time / Day", fontsize = 12)
    plt.ylabel("Altitude / km", fontsize = 12)
    plt.title(f"Altitude against Time (Monte Carlo, {config.runs} runs)", fontsize = 14)
    
    plt.legend()
    plt.show()

def save_pix(): #if required
    plt.savefig("orbit.png", dpi = 100)
    plt.savefig("altitude.png", dpi = 100)
    plt.savefig("omega.png", dpi = 100)
    plt.savefig("silhouette.png", dpi = 100)
    plt.savefig("stochastic.png", dpi = 100)
    plt.close()

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

def execute():
    start_time = time.perf_counter()
    config = Config()

    inertial_values_0 = coe_to_cartesian(config.a_0, config.e, config.theta_0, 
                                         config.omega_0, config.inc_0, config.raan_0, 
                                         config
    )
    quaternion_values_0 = config.q_0
    omega_values_0 = config.om_body_0

    state_0 = np.concatenate([inertial_values_0, quaternion_values_0, omega_values_0])

    print("")
    graph_type = input("Enter graph type: ")

    if graph_type in ["altitude", "orbit", "omega", "silhouette", "stochastic"]:
        print("Running simulation, graph type = "+ graph_type + "\n")
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
        state_array = result.y.T # T switches output as its (6, step) not needed (step, 6)

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
    elif graph_type == "omega":
        plot_omega(t_array, state_array, config)
    elif graph_type == "silhouette":
        plot_silhouette(t_array, state_array, config)
    elif graph_type == "stochastic":
        plot_stochastic(t_array_ref, all_altitude_arrays, config)
    else:
        raise ValueError("Invalid graph type.")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")      

# run simulation
os.system('cls' if os.name == 'nt' else 'clear')
execute()

