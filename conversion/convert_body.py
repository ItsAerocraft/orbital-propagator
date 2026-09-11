import numpy as np

from tumble.quaternion import quaternion_conjugate, quaternion_multiply


def inertial_to_body(inertial_vector, q):
    '''
    Converts a 3D Cartesian vector from the inertial frame to the object's body frame.

    Args:
        [inertial_vector]: 3-element array (e.g. position, velocity) in the inertial frame.
        [q]: Unit quaternion (unitless) representing the object's current orientation.

    Returns:
        3-element array representing the same vector in the body frame.

    Uses the rotation [q]* ⊗ [inertial_vector] ⊗ [q] (non-commutative quaternion
    multiplication), where [q]* is the conjugate of [q].
    '''

    q_inertial = np.concatenate([[0], inertial_vector])
    q_conjugated = quaternion_conjugate(q)

    multiply_1 = quaternion_multiply(q_conjugated, q_inertial)
    multiply_2 = quaternion_multiply(multiply_1, q)

    return multiply_2[1:4]
