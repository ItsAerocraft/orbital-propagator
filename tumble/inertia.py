import numpy as np

def inertia_cylinder(mass, radius, height):
    '''
    Function takes attributes of the cylinder as inputs.
    Function computes the resolved moment of inertia values in x, y, z directions.
    Function returns inertia matrix for object, to be used to calculate gravity-gradient torque.
    '''

    #standard formulae for moment of inertia on each axis for cylinder
    I_zz = 0.5 * mass * (radius ** 2)
    I_xx = I_yy = (1 / 12) * mass * (3 * (radius ** 2) + (height ** 2))

    #due to body's axis aligned with object's axis of rotation, use diagonal inertia matrix for simplification
    return np.array([[I_xx, 0, 0],
                    [0, I_yy, 0],
                    [0, 0, I_zz]
                    ])
    

def inertia_cuboid(mass, length, width, height):
    '''
    Function takes attributes of the cuboid as inputs.
    Function computes the resolved moment of inertia values in x, y, z directions.
    Function returns inertia matrix for object, to be used to calculate gravity-gradient torque.
    '''

    #standard formulae for moment of i nertia on each axis for cuboid
    const = mass / 12
    I_xx = const * ((width ** 2) + (height ** 2))
    I_yy = const * ((length ** 2) + (height ** 2))
    I_zz = const * ((length ** 2) + (width ** 2)) 

    #same reasoning as above
    return np.array([[I_xx, 0, 0],
                    [0, I_yy, 0],
                    [0, 0, I_zz]
                    ])

