import numpy as np

from forces.force_gravity import gravity_accel
from forces.force_drag import drag_accel
from forces.force_perturbation import perturbation_accel

from torques.torque_gravitygradient import gravgrad_torque
from torques.torque_aerodynamic import aerod_torque

from tumble.quaternion import quaternion_derivative
from tumble.rotation import angular_acceleration


def dynamics(t, state, config):
    '''
    Computes the full state derivative vector for the coupled orbital and
    attitude dynamics.

    Args:
        [t]: Elapsed time since epoch (s).
        [state]: 13-element state vector: elements 0:3 are ECI position (m),
            3:6 are ECI velocity (m s^-1), 6:10 are the orientation quaternion
            (unitless), and 10:13 are the body-frame angular velocity (rad s^-1).
        [config]: Simulation config, passed through to the gravity, drag,
            perturbation, torque, and rotation functions.

    Returns:
        13-element array of derivatives, ordered to match [state]: velocity
        (m s^-1), acceleration (m s^-2), quaternion rate (s^-1), and angular
        acceleration (rad s^-2).

    Called repeatedly by the adaptive solve_ivp integrator to build the state
    trajectory over 2000 adaptive time-steps.
    '''

    vel_vctr = state[3:6]
    q = state[6:10]
    omega = state[10:13]

    sum_accel = gravity_accel(t, state, config) + drag_accel(t, state, config) + perturbation_accel(t, state, config)

    q_dot = quaternion_derivative(q, omega)

    torque = gravgrad_torque(t, state, config) + aerod_torque(t, state, config)
    om_dot = angular_acceleration(omega, torque, config)

    return np.concatenate([vel_vctr, sum_accel, q_dot, om_dot])
