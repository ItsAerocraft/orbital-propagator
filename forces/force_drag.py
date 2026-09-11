import numpy as np
from environment.atmosphere import density
from tumble.area import projected_area_cylinder, projected_area_cuboid


def drag_accel(t, state, config):
    '''
    Computes the 3D acceleration vector due to atmospheric drag, opposing the object's velocity.

    Args:
        [t]: Elapsed time since epoch (s).
        [state]: State vector whose elements 3:6 are the ECI velocity (m s^-1).
        [config]: Simulation config, providing [object] (shape flag), [Cd]
            (drag coefficient, unitless), and [m] (mass, kg).

    Returns:
        3-element array representing the drag acceleration vector (m s^-2).

    Uses the standard drag equation, assuming a constant drag coefficient and
    center of pressure: a = -0.5 * rho * [Cd] * [ref_area] * v / [m], directed
    opposite the velocity vector.
    '''

    rho = density(t, state, config)

    vel_vctr = np.array(state[3:6])
    mag_v = np.linalg.norm(vel_vctr)

    if config.object == "cylinder":
        ref_area = projected_area_cylinder(state, config)
        drag_factor = ((- 0.5 * (rho * config.Cd * ref_area)) / config.m) * mag_v

    elif config.object == "cuboid":
        ref_area = projected_area_cuboid(state, config)
        drag_factor = ((- 0.5 * (rho * config.Cd * ref_area)) / config.m) * mag_v

    return drag_factor * vel_vctr
