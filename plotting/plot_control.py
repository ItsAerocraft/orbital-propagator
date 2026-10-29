import numpy as np

import matplotlib.pyplot as plt

from plotting.style import minorgridlines

from z_main.config import TAU_MAX


def plot_error(results, t_max = None):
    '''
    Plots each controller's attitude error angle against time on shared axes.

    Args:
        [results]: Dict mapping controller name to its run result, each
            providing [t] (s) and [error_deg] (deg).
        [t_max]: Upper time limit of the plot (s); None shows the full run.
    '''
    plt.figure(figsize = (10, 6))

    for name, run in results.items():
        plt.plot(run["t"], run["error_deg"], label = name)

    if t_max is not None:
        plt.xlim(0, t_max)

    minorgridlines()

    plt.xlabel("Time / s", fontsize = 12)
    plt.ylabel("Attitude Error / deg", fontsize = 12)
    plt.title("Attitude Error against Time", fontsize = 14)

    plt.legend()
    plt.show()


def plot_torque(results, tau_max, t_max = None):
    '''
    Plots each controller's largest control torque component against time on
    shared axes, with a dashed line at the actuator limit.

    Args:
        [results]: Dict mapping controller name to its run result, each
            providing [t] (s) and [torque] (N x 3 array, N m).
        [tau_max]: Actuator maximum torque limit per component (N m).
        [t_max]: Upper time limit of the plot (s); None shows the full run.

    The largest component (infinity norm) is plotted because the actuator limit
    clips each component separately, so it is the quantity [tau_max] bounds.
    '''
    plt.figure(figsize = (10, 6))

    for name, run in results.items():
        plt.plot(run["t"], np.max(np.abs(run["torque"]), axis = 1), label = name)

    plt.axhline(tau_max, color = "black", linestyle = "--", label = r"$\tau_{max}$")

    if t_max is not None:
        plt.xlim(0, t_max)

    minorgridlines()

    plt.xlabel("Time / s", fontsize = 12)
    plt.ylabel(r"Control Torque $|\tau|_\infty$ / N m", fontsize = 12)
    plt.title("Control Torque against Time", fontsize = 14)

    plt.legend()
    plt.show()


def plot_audit(t_array, margins):
    '''
    Plots the contraction audit margin against time.

    Args:
        [t_array]: Array of sample times (s).
        [margins]: Array of the largest eigenvalue of C(x) at each sample;
            the contraction condition holds where it is <= 0.
    '''
    plt.figure(figsize = (10, 6))
    plt.plot(t_array, margins, label = "margin")

    plt.axhline(0, color = "black", linestyle = "--", label = "limit (holds below)")

    minorgridlines()

    plt.xlabel("Time / s", fontsize = 12)
    plt.ylabel(r"Max eigenvalue of $C(x)$", fontsize = 12)
    plt.title("Contraction Audit Margin against Time", fontsize = 14)

    plt.legend()
    plt.show()


def plot_comparison(results, margins):
    '''
    Plots the controller comparison: error and torque zoomed in on the settling
    transient, and the contraction audit margin over the full window.

    Args:
        [results]: Dict mapping controller name to its run result.
        [margins]: Contraction audit margin at each sample.

    The zoom window is twice the slowest settling time, so the approach to zero
    error is visible; if any controller did not settle, the full window is shown.
    '''
    settling_times = [run["settling_time"] for run in results.values()]

    if any(t is None for t in settling_times) or max(settling_times) == 0:
        t_zoom = None
    else:
        t_zoom = 2 * max(settling_times)

    plot_error(results, t_zoom)
    plot_torque(results, TAU_MAX, t_zoom)
    plot_audit(results["contraction"]["t"], margins)
