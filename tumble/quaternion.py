import numpy as np

def quaternion_derivative(q, omega):
    """
    Function takes a quaternion and angular velocity (at that instant in time) as input.
    Function computes the derivative of the quaternion at that instant in time.
    """

    q0, q1, q2, q3 = q #extract quaternion components from input
    om_x, om_y, om_z = omega #extract angular velocity components from input

    #derived relationship between derivatives of quaternion components and angular velocity components
    q0_dot = - 0.5 * ((q1 * om_x) + (q2 * om_y) + (q3 * om_z))
    q1_dot = 0.5 * ((q0 * om_x) + (q2 * om_z) - (q3 * om_y))
    q2_dot = 0.5 * ((q0 * om_y) + (q3 * om_x) - (q1 * om_z))
    q3_dot = 0.5 * ((q0 * om_z) + (q1 * om_y) - (q2 * om_x))

    return np.array([q0_dot, q1_dot, q2_dot, q3_dot]) #returns set of quaternion derivatives

def quaternion_conjugate(q):
    """
    Function takes an imaginary quaternion as input.
    Function computes the conjugate of the quaternion. 
    """

    q0, q1, q2, q3 = q

    return np.array([q0, - q1, -q2, -q3]) #conjugation equal to multiply by [1, -1, -1, -1] if quaternion is pure (only imaginary)

def quaternion_multiply(q_a, q_b):
    """
    Function takes 2 quaternions as input.
    Function computes the noncommutative result of the multiplication (q_a ⊗ q_b)
    """
    
    #extract quaternion components from input
    q_a0, q_a1, q_a2, q_a3 = q_a 
    q_b0, q_b1, q_b2, q_b3 = q_b

    #each quaternion's imaginary vector
    q_a_vector = np.array([q_a1, q_a2, q_a3]) 
    q_b_vector = np.array([q_b1, q_b2, q_b3])

    #computes scalar and vector parts resulting from quaternion multiplication
    scalar = np.array([(q_a0 * q_b0) - np.dot(q_a_vector, q_b_vector)])
    vector = (q_a0 * q_b_vector) + (q_b0 * q_a_vector) + np.cross(q_a_vector, q_b_vector)

    return np.concatenate([scalar, vector]) #returns 4-element quaternion
