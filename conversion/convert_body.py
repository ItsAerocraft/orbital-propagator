import numpy as np

from tumble.quaternion import quaternion_conjugate, quaternion_multiply

def inertial_to_body(inertial_vector, q):
    '''
    Takes the inertial vector, the array representing the 3D cartesian vector (e.g. position, velocity, etc) of the object (input 1).
    Takes a quaternion vector that represents the rotation of the object (input 2).
    This converts the reference frame from the inertial frame to the body's reference frame.
    '''

    q_inertial = np.concatenate([[0], inertial_vector]) # concatenates the real part of 0 forming a pure quaternion
    q_conjugated = quaternion_conjugate(q) # conjugates the rotation quaternion, necessary for rotation of the body

    #inertial to body order:  q* ⊗ v ⊗ q (non commutative)
    multiply_1 = quaternion_multiply(q_conjugated, q_inertial) # stores result of multiplying of q_conjugated by q_inertial
    multiply_2 = quaternion_multiply(multiply_1, q) # multiplies the above result by q

    return multiply_2[1:4] #results in pure quaternion, so return vector values only for final 3D cartesian vector for body frame
