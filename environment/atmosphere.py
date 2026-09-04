import numpy as np
from pymsis import msis
from datetime import timedelta #needed to add to epoch time

def cartesian_geodetic(t, state, config):
    long_unfix = np.arctan2(state[1], state[0]) # y, x
    #atan2 returns the actual angle without getting spoofed by double answers
    long_rad = long_unfix - (config.omega_earth * t)

    pos_vctr = np.array(state[0:3]) # extract items x, y, z from state vector
    mag_r = np.linalg.norm(pos_vctr) # obtain magnitude of position vector
    lat_rad = np.arcsin(state[2] / mag_r)

    alt = mag_r - config.R

    return(long_rad, lat_rad, alt) #tuple rather than array because its 3 heterogenous quantities

def density(t, state, config):
    long_rad, lat_rad, alt = cartesian_geodetic(t, state, config)
    long_deg, lat_deg, alt_km = np.rad2deg(long_rad), np.rad2deg(lat_rad), (alt / 1000) #msis required units

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


