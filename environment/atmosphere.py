import numpy as np
from pymsis import msis
from datetime import timedelta


def cartesian_geodetic(t, state, config):
    '''
    Converts an ECI Cartesian state into geodetic longitude, latitude, and altitude.

    Args:
        [t]: Elapsed time since epoch (s).
        [state]: State vector whose first three elements are the ECI position (m).
        [config]: Simulation config, providing [omega_earth] (rad s^-1) and [R] (m).

    Returns:
        Tuple ([long_rad], [lat_rad], [alt]) of longitude (rad), latitude (rad), and altitude (m).

    Longitude is corrected for Earth's rotation since epoch to convert from
    the inertial frame into an Earth-fixed frame.
    '''

    long_unfix = np.arctan2(state[1], state[0])
    long_rad = long_unfix - (config.omega_earth * t)

    pos_vctr = np.array(state[0:3])
    mag_r = np.linalg.norm(pos_vctr)
    lat_rad = np.arcsin(state[2] / mag_r)

    alt = mag_r - config.R

    return (long_rad, lat_rad, alt)


def density(t, state, config):
    '''
    Computes atmospheric mass density at the object's position using NRLMSIS.

    Args:
        [t]: Elapsed time since epoch (s).
        [state]: State vector whose first three elements are the ECI position (m).
        [config]: Simulation config, providing [epoch], [stochastic] (flag),
            [f107_trajectory] (used when [stochastic] is True), [f107_mean]
            (used when [stochastic] is False), and [Ap].

    Returns:
        Atmospheric mass density at the given time and position (kg m^-3).
    '''

    long_rad, lat_rad, alt = cartesian_geodetic(t, state, config)
    long_deg, lat_deg, alt_km = np.rad2deg(long_rad), np.rad2deg(lat_rad), (alt / 1000)  # pymsis expects degrees and km

    if config.stochastic == True:
        t_samples, f107_samples = config.f107_trajectory
        f107_current = np.interp(t, t_samples, f107_samples)
 
        result = msis.run(
            config.epoch + timedelta(seconds = t),
            long_deg,
            lat_deg,
            alt_km,
            f107s = f107_current,
            f107as = f107_current,
            aps = [[config.Ap] * 7]
        )

    else:
        result = msis.run(
                    config.epoch + timedelta(seconds = t),
                    long_deg,
                    lat_deg,
                    alt_km,
                    f107s = config.f107_mean,
                    f107as = config.f107_mean,
                    aps = [[config.Ap] * 7]
                )

    return result[0, 0]


