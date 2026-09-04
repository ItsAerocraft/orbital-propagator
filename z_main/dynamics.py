import numpy as np

from forces.force_gravity import gravity_accel
from forces.force_drag import drag_accel
from forces.force_perturbation import perturbation_accel

from torques.torque_gravitygradient import gravgrad_torque
from torques.torque_aerodynamic import aerod_torque

from tumble.quaternion import quaternion_derivative
from tumble.rotation import angular_acceleration

def dynamics(t, state, config):
    """
    Function takes state vector and time as inputs.
    Function computes the set of derivatives (of displacement, of velocity, of set of quaternions, of angular velocity) in 3D space.
    Function is looped through adaptive solve_ivp to generate array of state vectors for each of 2000 adaptive time-steps.
    """

    velocity = state[3:6]
    q = state[6:10]
    omega = state[10:13]

    #sum of acceleration from gravitational force, drag force, and effect of perturbation factors (J2, J3, J4)
    sum_accel = gravity_accel(t, state, config) + drag_accel(t, state, config) + perturbation_accel(t, state, config) 

    #obtain derivatives for set of quaternions
    q_dot = quaternion_derivative(q, omega)

    #sum of gravity-gradient and aerodynamic torque, then computes resultant angular acceleration
    torque = gravgrad_torque(t, state, config) + aerod_torque(t, state, config)
    om_dot = angular_acceleration(omega, torque, config)

    return np.concatenate([velocity, sum_accel, q_dot, om_dot]) #returns array of derivatives at correct positions corresponding to antiderivatives

