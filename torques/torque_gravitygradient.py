import numpy as np

from conversion.convert_body import inertial_to_body

from tumble.inertia import inertia_cuboid, inertia_cylinder

def gravgrad_torque(t, state, config):
    '''
    Function takes state vector and time as inputs.
    Function computes and returns the gravity gradient torque using the state vector.
    The returned function affects angular velocity components of state vector.
    '''

    #extraction from state vector
    x, y, z = state[0:3]
    q = state[6:10]
    mag_r = np.linalg.norm(state[0:3]) 

    r_hat_inertial = state[0:3] / mag_r #3D inertial position vector for the object
    r_hat_body = inertial_to_body(r_hat_inertial, q) #rotates the inertial position vector into the body frame

    #the following uses the function inertia_cylinder to compmute and return the moment of inertia 3 X 3 matrices, depending on the object type
    if config.object == "cylinder":
        mass, radius, height = config.m, config.radius_cylinder, config.height_cylinder
        I = inertia_cylinder(mass, radius, height)

    elif config.object == "cuboid":
        mass, length, width, height = config.m, config.length_cuboid, config.width_cuboid, config.height_cuboid
        I = inertia_cuboid(mass, length, width, height)

    I_mult_r_hat = I @ r_hat_body #multiplies the moment of inertia matrix with the body frame/rotated position vector

    result = ((3 * config.mu) / (mag_r ** 3)) * np.cross(r_hat_body,I_mult_r_hat) #standard formula for gravity gradient torque

    return result
