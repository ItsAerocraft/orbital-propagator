import numpy as np


def inertia_cylinder(mass, radius, height):
    '''
    Computes the diagonal moment of inertia matrix for a cylindrical object.

    Args:
        [mass]: Object mass (kg).
        [radius]: Cylinder radius (m).
        [height]: Cylinder height (m).

    Returns:
        3x3 array representing the moment of inertia matrix (kg m^2).

    The body frame is aligned with the object's axis of rotation, so the
    off-diagonal terms are zero and the matrix reduces to the diagonal case.
    '''

    I_zz = 0.5 * mass * (radius ** 2)
    I_xx = I_yy = (1 / 12) * mass * (3 * (radius ** 2) + (height ** 2))

    return np.array([[I_xx, 0, 0],
                    [0, I_yy, 0],
                    [0, 0, I_zz]
                    ])


def inertia_cuboid(mass, length, width, height):
    '''
    Computes the diagonal moment of inertia matrix for a cuboid object.

    Args:
        [mass]: Object mass (kg).
        [length]: Cuboid length (m).
        [width]: Cuboid width (m).
        [height]: Cuboid height (m).

    Returns:
        3x3 array representing the moment of inertia matrix (kg m^2).

    The body frame is aligned with the object's axis of rotation, so the
    off-diagonal terms are zero and the matrix reduces to the diagonal case.
    '''

    const = mass / 12
    I_xx = const * ((width ** 2) + (height ** 2))
    I_yy = const * ((length ** 2) + (height ** 2))
    I_zz = const * ((length ** 2) + (width ** 2))

    return np.array([[I_xx, 0, 0],
                    [0, I_yy, 0],
                    [0, 0, I_zz]
                    ])


def inertia(config):
    '''
    Computes the diagonal moment of inertia matrix for the configured object shape.

    Args:
        [config]: Simulation config, providing [object] (shape flag), [m]
            (mass, kg), and the relevant dimensions: [radius_cylinder] (m),
            [height_cylinder] (m) for a cylinder, or [length_cuboid] (m),
            [width_cuboid] (m), [height_cuboid] (m) for a cuboid.

    Returns:
        3x3 array representing the moment of inertia matrix (kg m^2).

    Raises:
        ValueError: If [config.object] is not a supported shape.
    '''

    if config.object == "cylinder":
        return inertia_cylinder(config.m, config.radius_cylinder, config.height_cylinder)

    elif config.object == "cuboid":
        return inertia_cuboid(config.m, config.length_cuboid, config.width_cuboid, config.height_cuboid)

    else:
        raise ValueError(f"Unknown object shape: {config.object!r} (expected 'cylinder' or 'cuboid').")
