import numpy as np
import datetime

from dataclasses import dataclass, field


@dataclass
class Config:
    # GENERAL
    epoch: datetime.datetime = datetime.datetime(2026, 1, 1)

    # PLANET
    R: float = 6.378e6  # (m) [radius of earth]
    G: float = 6.674e-11  # (m^3 kg^-1 s^-2) [gravitational constant]
    M: float = 5.972e24  # (kg) [mass of earth]

    mu: float = M * G  # (m^3 s^-2) [standard gravitational parameter, G*M]

    omega_earth: float = 7.292e-5  # (rad s^-1) [angular velocity of earth's rotation]


    # ELLIPSE
    r_peri: float = R + 220000  # (m) [distance between periapsis and center of earth]
    r_apo: float = R + 750000  # (m) [distance between apoapsis and center of earth]

    a_0: float = (r_apo + r_peri) / 2  # (m) [initial semi-major axis, classical orbital element]
    e: float = (r_apo - r_peri) / (r_apo + r_peri)  # (unitless) [eccentricity of ellipse, classical orbital element]

    theta_0: float = 0  # (rad) [true anomaly, classical orbital element]
    omega_0: float = np.pi / 6  # (rad) [argument of periapsis, classical orbital element]
    inc_0: float = np.pi / 6  # (rad) [inclination angle, classical orbital element]
    raan_0: float = np.pi / 3  # (rad) [right ascension of the ascending node, measured from vernal equinox, classical orbital element]

    T_orbit: float = 2 * np.pi * (((a_0 ** 3) / mu) ** 0.5)  # (s) [initial period of orbit]

    # SATELLITE
    m: float = 1000  # (kg) [mass of satellite]
    Cd: float = 2.2  # (unitless) [drag coefficient]

    radius_cylinder: float = 1.0  # (m) [radius of cylindrical satellite]
    height_cylinder: float = 3.0  # (m) [height of cylindrical satellite]

    length_cuboid: float = 5.0  # (m) [length of cuboid satellite]
    width_cuboid: float = 2.0  # (m) [width of cuboid satellite]
    height_cuboid: float = 0.5  # (m) [height of cuboid satellite]

    # TUMBLE
    r_cp: tuple = (0.05, 0.02, 0.01)  # (m) [vector from center of mass to center of pressure, approximated]

    q_0: tuple = (1, 0, 0, 0)  # (unitless) [initial quaternion of turn (q0) and axis (q1, q2, q3)]
    om_body_0: tuple = (0.01, 0, 0)  # (rad s^-1) [initial angular velocity components]


    # ATMOSPHERE
    stochastic: bool = False  # [whether Monte-Carlo mode is chosen]

    f107_mean: float = 150.0  # (sfu) [mean solar radio flux value]
    reversion_strength: float = 1e-6  # (day^-1) [pull strength to return value to mean]
    volatility: float = 0.5  # (sfu day^-0.5) [size of random shocks to values]

    f107_trajectory: tuple = None  # [tuple of (t_samples, f107_samples) arrays]

    dt_stochastic: float = 86400  # (s) [step time for stochastic output]

    Ap: float = 10.0  # (unitless) [geomagnetic Ap index value(s)]

    # PERTURBATION
    J2: float = 1.082e-3  # (unitless) [constant for J2 / primary equatorial oblateness]
    J3: float = -2.532e-6  # (unitless) [constant for J3 / north-south asymmetric pear shape]
    J4: float = -1.611e-6  # (unitless) [constant for J4 / symmetric flattening of polar regions]


    # INTEGRATION
    days: float = 100  # (days) [number of days]
    T: float = days * 24 * 3600  # (s) [total duration for integration]

    # SELECTION
    object: str = "cuboid"  # (cylinder / cuboid) [chosen shape of satellite]
    runs: int = 15  # (integer) [number of runs for Monte-Carlo simulation, if chosen]
