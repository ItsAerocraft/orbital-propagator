import numpy as np
from conversion.convert_body import inertial_to_body


def projected_area_cylinder(state, config):
    '''
    Computes the cylinder's silhouette area facing the velocity direction,
    used as the reference area for drag force.

    Args:
        [state]: State vector whose elements 3:6 are the ECI velocity (m s^-1)
            and 6:10 the orientation quaternion (unitless).
        [config]: Simulation config, providing [radius_cylinder] (m) and
            [height_cylinder] (m).

    Returns:
        Silhouette area of the cylinder facing the velocity direction (m^2).

    The silhouette is the sum of the circular base's projection and the
    rectangular side's projection, weighted by the cosine/sine of the angle
    between the body z-axis and the velocity vector (in the body frame).
    '''
    vel_vctr = state[3:6]
    mag_v = np.linalg.norm(vel_vctr)
    v_hat_inertial = vel_vctr / mag_v

    q = state[6:10]
    v_hat_body = inertial_to_body(v_hat_inertial, q)

    z_axis_hat = np.array([0, 0, 1])
    cos_angle = np.dot(v_hat_body, z_axis_hat)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # guards against floating-point error pushing |cos_angle| above 1

    sin_angle = abs(np.sqrt(1 - (cos_angle ** 2)))

    silhouette = (np.pi * (config.radius_cylinder ** 2) * cos_angle) + (2 * config.radius_cylinder * config.height_cylinder * sin_angle)

    return silhouette


def projected_area_cuboid(state, config):
    '''
    Computes the cuboid's silhouette area facing the velocity direction,
    used as the reference area for drag force.

    Args:
        [state]: State vector whose elements 3:6 are the ECI velocity (m s^-1)
            and 6:10 the orientation quaternion (unitless).
        [config]: Simulation config, providing [length_cuboid] (m),
            [width_cuboid] (m), and [height_cuboid] (m).

    Returns:
        Silhouette area of the cuboid facing the velocity direction (m^2).

    Each face's contribution is its area weighted by the cosine of the angle
    between its normal (a body axis) and the velocity vector, i.e. the
    body-frame velocity component along that axis; only the (up to three)
    faces facing the direction of motion contribute.
    '''
    vel_vctr = state[3:6]
    mag_v = np.linalg.norm(vel_vctr)
    v_hat_inertial = vel_vctr / mag_v

    q = state[6:10]
    v_hat_body = inertial_to_body(v_hat_inertial, q)

    vx, vy, vz = v_hat_body

    x_face = abs(vx) * config.width_cuboid * config.height_cuboid
    y_face = abs(vy) * config.length_cuboid * config.height_cuboid
    z_face = abs(vz) * config.length_cuboid * config.width_cuboid

    silhouette = x_face + y_face + z_face
    return silhouette
