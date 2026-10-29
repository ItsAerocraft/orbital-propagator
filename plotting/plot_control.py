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


def plot_audit(t_array, margins, clipped):
    '''
    Plots the contraction audit margin against time, shading the intervals
    where the torque clipped, since the certificate does not cover them.

    Args:
        [t_array]: Array of sample times (s).
        [margins]: Array of the largest eigenvalue of C(x) at each sample;
            the contraction condition holds where it is <= 0.
        [clipped]: Array of bools, True where the commanded torque exceeded
            the actuator limit.
    '''
    plt.figure(figsize = (10, 6))
    plt.plot(t_array, margins, label = "margin")

    # contiguous runs of clipped samples, as (start, end) index pairs
    edges = np.diff(np.concatenate([[0], clipped.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1

    for i, (start, end) in enumerate(zip(starts, ends)):
        plt.axvspan(t_array[start], t_array[end], color = "grey", alpha = 0.3,
                    label = "clipped, not covered" if i == 0 else None)

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
        t_zoom = min(2 * max(settling_times), max(run["t"][-1] for run in results.values()))

    plot_error(results, t_zoom)
    plot_torque(results, TAU_MAX, t_zoom)
    plot_audit(results["contraction"]["t"], margins, results["contraction"]["clipped"])


def plot_tradeoff(sweep, metric, ylabel, title, did_not_settle = None):
    '''
    Plots one metric against control effort for every swept gain, one curve
    per controller, on log axes.

    Args:
        [sweep]: Result of [control.sweep.run_sweep].
        [metric]: Key of the metric to plot on the y axis.
        [ylabel], [title]: Axis label and title.
        [did_not_settle]: y value at which to draw runs whose metric is None
            (hollow markers); None leaves them out.
    '''
    plt.figure(figsize = (10, 6))

    for name, label_key, marker in [("pd", "omega_n", "o"), ("contraction", "lam", "s")]:
        points = sorted(sweep[name], key = lambda p: p["effort"])

        done = [p for p in points if p[metric] is not None]
        line, = plt.plot([p["effort"] for p in done], [p[metric] for p in done], marker = marker, label = name)

        for p in done:
            plt.annotate(f"{p[label_key]:g}", (p["effort"], p[metric]), textcoords = "offset points",
                         xytext = (4, 4), fontsize = 8, color = line.get_color())

        missing = [p for p in points if p[metric] is None]

        if did_not_settle is not None and missing:
            plt.scatter([p["effort"] for p in missing], [did_not_settle] * len(missing), marker = marker,
                        facecolors = "none", edgecolors = line.get_color(), label = f"{name} (did not settle)")

    plt.xscale("log")
    plt.yscale("log")

    minorgridlines()

    plt.xlabel(r"Control Effort $\int |\tau|^2 dt$ / N$^2$ m$^2$ s", fontsize = 12)
    plt.ylabel(ylabel, fontsize = 12)
    plt.title(title + r" (labels: $\omega_n$ for pd, $\lambda$ for contraction)", fontsize = 14)

    plt.legend()
    plt.show()


def plot_sweep(sweep, duration):
    '''
    Plots steady-state offset and settling time against control effort for
    the PD and contraction gain sweeps.

    Args:
        [sweep]: Result of [control.sweep.run_sweep].
        [duration]: Simulated window (s); runs that did not settle are drawn there.
    '''
    plot_tradeoff(sweep, "steady_state_error", "Steady-State Offset / deg", "Offset against Effort")
    plot_tradeoff(sweep, "settling_time", "Settling Time / s", "Settling Time against Effort", did_not_settle = duration)
