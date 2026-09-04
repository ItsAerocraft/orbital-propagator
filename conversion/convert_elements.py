import numpy as np

def R_x(angle):
    '''
    Generates 3 X 3 coordinate rotation matrices to rotate a 3D vector around the x-axis by a given angle (input).
    Angles are measured in radians.
    '''
    
    matrix_rotate_x = np.array([[1, 0, 0], # 1 in first row, x position to guarentee that x remains the same value (rotate around x)
                   [0, np.cos(angle), - np.sin(angle)], # standard rotation of axis formulae
                   [0, np.sin(angle), np.cos(angle)]
                   ])

    return matrix_rotate_x

def R_z(angle):
    '''
    Same function as R_x above, but for z-axis.
    '''
    
    matrix_rotate_z = np.array([[np.cos(angle), - np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1] # same concept as rotate around x-axis above, by keeping z at the same value
                   ])

    return matrix_rotate_z

def coe_to_cartesian(a, e, theta, omega, inc, raan, config):
    '''
    Takes classical orbital elements as inputs.
    Function uses rotational matrices by multiplying them to the 3D position and velocity vectors of the object (in perifocal frame).
    Function returns rotated position and velocity vectors of the object in cartesian form.
    
    '''

    p = a * (1 - (e ** 2)) # semi-latus rectum, drawn from foci to conic section, perpendicular to semi-major axis
    h = np.sqrt(config.mu * p) # specific angular momentum

    r = p / (1 + e * np.cos(theta)) # initial distance between object and planet

    # x-axis aligned with vernal equinox, y-axis perpendicular to x-axis flat against equatorial plane, thus z = 0
    r_pf = np.array([r * np.cos(theta), r * np.sin(theta), 0]) # calculates radius vector of object in perifocal frame

    v_pf = np.array([ - (h / p) * np.sin(theta),  (h / p) * (e + np.cos(theta)), 0]) #calculates velocity vector of object in perifocal frame

    rotate_matrix = R_z(raan) @ R_x(inc) @ R_z(omega) # conglomerates rotational matrices by multiplying them non-commutatively
    # rotate around z-axis by Argument of Perigee (omega)
    # then around x-axis by Inclination Angle (inc), 
    # then finally around new z-axis (was rotated by second step) by Right Ascension of the Ascending Node (raan)

    r_final = rotate_matrix @ r_pf # final multiplication of rotational matrices for the object in perifocal frame
    v_final = rotate_matrix @ v_pf

    return np.concatenate([r_final, v_final])

def current_semimajor_axis(state, config):
    r = np.linalg.norm(state[0:3])
    v = np.linalg.norm(state[3:6])
    subtraction = (config.mu / r) - ((v ** 2) / 2)

    a = config.mu / (2 * subtraction)

    return a

def current_orbital_period(state, config):
    a = current_semimajor_axis(state, config)
    period = 2 * np.pi * np.sqrt((a ** 3) / config.mu)

    return period




