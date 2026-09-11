import numpy as np


def generate_f107_trajectory(config, seed = None):
    '''
    Generates a stochastic F10.7 solar flux trajectory using an
    Ornstein-Uhlenbeck (mean-reverting) random walk.

    Args:
        [config]: Simulation config, providing [dt_stochastic] (s), [T] (s),
            [f107_mean] (sfu), [reversion_strength] (day^-1), and [volatility]
            (sfu day^-0.5).
        [seed]: Optional random seed for reproducibility.

    Returns:
        Tuple ([t_samples], [f107_samples]) of sample times (s) and the
        corresponding F10.7 values (sfu).

    The reversion and volatility terms use [dt] expressed in days, since
    [reversion_strength] and [volatility] are defined per day.
    '''
    rng = np.random.default_rng(seed)
    dt = config.dt_stochastic
    dt_days = dt / 86400

    steps = int(config.T / dt)

    t_samples = np.zeros(steps + 1)
    f107_samples = np.zeros(steps + 1)

    t_samples[0] = 0
    f107_samples[0] = config.f107_mean

    for i in range(1, steps + 1):
        t_samples[i] = t_samples[i - 1] + dt

        dW = rng.normal(0, np.sqrt(dt_days))
        dF = config.reversion_strength * (config.f107_mean - f107_samples[i - 1]) * dt_days + config.volatility * dW

        f107_samples[i] = f107_samples[i - 1] + dF

    return t_samples, f107_samples
