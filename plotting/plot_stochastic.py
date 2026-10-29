import numpy as np

import matplotlib.pyplot as plt

from plotting.style import minorgridlines


def plot_stochastic(t_array_ref, all_altitude_arrays, config):
    '''
    Plots the min/max/average perigee altitude envelope across Monte Carlo runs.

    Args:
        [t_array_ref]: Reference array of sample times (s).
        [all_altitude_arrays]: Array of per-run perigee altitude arrays (m),
            interpolated onto [t_array_ref] (NaN outside a run's data range).
        [config]: Simulation config, providing [runs] (number of Monte Carlo runs).
    '''
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
