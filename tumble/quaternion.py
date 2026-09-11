import numpy as np


def quaternion_derivative(q, omega):
    '''
    Computes the time derivative of a quaternion given the current angular velocity.

    Args:
        [q]: 4-element quaternion (unitless), ([q0], [q1], [q2], [q3]).
        [omega]: 3-element body-frame angular velocity vector (rad s^-1).

    Returns:
        4-element array representing the quaternion derivative (s^-1).

    Derived from the standard relationship between quaternion component
    derivatives and angular velocity components.
    '''

    q0, q1, q2, q3 = q
    om_x, om_y, om_z = omega

    q0_dot = - 0.5 * ((q1 * om_x) + (q2 * om_y) + (q3 * om_z))
    q1_dot = 0.5 * ((q0 * om_x) + (q2 * om_z) - (q3 * om_y))
    q2_dot = 0.5 * ((q0 * om_y) + (q3 * om_x) - (q1 * om_z))
    q3_dot = 0.5 * ((q0 * om_z) + (q1 * om_y) - (q2 * om_x))

    return np.array([q0_dot, q1_dot, q2_dot, q3_dot])


def quaternion_conjugate(q):
    '''
    Computes the conjugate of a quaternion.

    Args:
        [q]: 4-element quaternion (unitless), ([q0], [q1], [q2], [q3]).

    Returns:
        4-element array representing the conjugated quaternion (unitless).

    For a pure (purely imaginary) quaternion, conjugation is equivalent to
    multiplying by [1, -1, -1, -1].
    '''

    q0, q1, q2, q3 = q

    return np.array([q0, - q1, -q2, -q3])


def quaternion_multiply(q_a, q_b):
    '''
    Computes the non-commutative product of two quaternions, [q_a] ⊗ [q_b].

    Args:
        [q_a]: 4-element quaternion (unitless).
        [q_b]: 4-element quaternion (unitless).

    Returns:
        4-element array representing the quaternion product [q_a] ⊗ [q_b] (unitless).
    '''

    q_a0, q_a1, q_a2, q_a3 = q_a
    q_b0, q_b1, q_b2, q_b3 = q_b

    q_a_vector = np.array([q_a1, q_a2, q_a3])
    q_b_vector = np.array([q_b1, q_b2, q_b3])

    scalar = np.array([(q_a0 * q_b0) - np.dot(q_a_vector, q_b_vector)])
    vector = (q_a0 * q_b_vector) + (q_b0 * q_a_vector) + np.cross(q_a_vector, q_b_vector)

    return np.concatenate([scalar, vector])
