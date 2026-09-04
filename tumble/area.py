import numpy as np
from conversion.convert_body import inertial_to_body

def projected_area_cylinder(t, state, config):
    '''
    Function takes state vector and time as inputs.
    Function computes the silhouette area (if object == cylinder in config) at this time step.
    The returned value gives the area of cylinder's silhouette to be used as reference area for drag force.
    '''
    velocity = state[3:6] 
    v_mag = np.linalg.norm(velocity)
    v_hat_inertial = velocity / v_mag #computes unit velocity vector 

    q = state[6:10]
    v_hat_body = inertial_to_body(v_hat_inertial, q) #converts inertial unit velocity vector into body frame

    z_axis_hat = np.array([0, 0, 1]) # z-axis vector
    cos_angle = np.dot(v_hat_body, z_axis_hat) # as both velocity vector and z-axis vector are unit length, dot product = cosine
    cos_angle = np.clip(cos_angle, -1.0, 1.0) # prevents error due to python floating point calculation error from previous line

    sin_angle = abs(np.sqrt(1 - (cos_angle ** 2)))

    #silhouette area of base + silhouette area of cylinder rectangular side (cross section) area depending on angle
    silhouette = (np.pi * (config.radius_cylinder ** 2) * cos_angle) + (2 * config.radius_cylinder * config.height_cylinder * sin_angle)

    return silhouette
    



def projected_area_cuboid(t, state, config):
    '''
    Function takes state vector and time as inputs.
    Function computes the silhouette area (if object == cuboid in config) at this time step.
    The returned value gives the area of cuboid's silhouette to be used as reference area for drag force.
    '''
    velocity = state[3:6] 
    v_mag = np.linalg.norm(velocity) 
    v_hat_inertial = velocity / v_mag

    q = state[6:10]
    v_hat_body = inertial_to_body(v_hat_inertial, q)

    vx, vy, vz = v_hat_body #extracts body frame velocity in x, y, z direction

    #unit velocity vector dotted with the unit axis vector just gives (1)(1)cos(angle) between them
    #example for x, unit x-axis = [1, 0, 0], x-axis dot with unit velocity vector just gives vx
    #two definitions for unit velocity vector dot with unit axis vector, which are equal to each other, thus |vx| = cos(angle)

    x_face = abs(vx) * config.width_cuboid * config.height_cuboid # respective cos(angle) multiplied with face/plane perpendicular to axis
    y_face = abs(vy) * config.length_cuboid * config.height_cuboid
    z_face = abs(vz) * config.length_cuboid * config.width_cuboid

    silhouette = x_face + y_face + z_face #sum of warped silhouette area of each visible face (only 3 at a time facing direction of motion)
    return silhouette


