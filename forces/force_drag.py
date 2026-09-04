import numpy as np
from environment.atmosphere import density
from tumble.area import projected_area_cylinder, projected_area_cuboid

def drag_accel(t, state, config):
    """
    Function returns 3D acceleration vector for drag opposing the objects velocity.
    """

    rho = density(t, state, config)

    vel_vctr = np.array(state[3:6])
    mag_v = np.linalg.norm(vel_vctr)

    #assuming constant drag coefficient and center of pressure, standard equation for drag f
    if config.object == "cylinder":
        ref_area = projected_area_cylinder(t, state, config)
        drag_factor = ((- 0.5 * (rho * config.Cd * ref_area)) / config.m) * mag_v

    elif config.object == "cuboid":
        ref_area = projected_area_cuboid(t, state, config)
        drag_factor = ((- 0.5 * (rho * config.Cd * ref_area)) / config.m) * mag_v


    return drag_factor * vel_vctr

