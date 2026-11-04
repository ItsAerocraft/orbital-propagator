import numpy as np

from z_main.config import KP, KD, TAU_MAX, Q_TARGET

from control.pd import pd_controller, pd_raw
from control.contraction import contraction_control, contraction_raw

from tumble.quaternion import quaternion_error_vec

def pd_gains(config):
    '''
    Returns the PD gains ([Kp], [Kd]), taking the config overrides when set.

    Args:
        [config]: Simulation config, providing [kp] and [kd] overrides (None
            to use the defaults KP and KD).

    Returns:
        Tuple of ([Kp], [Kd]): proportional gain (N m) and derivative gain (N m s).
    '''
    kp = KP if config.kp is None else config.kp
    kd = KD if config.kd is None else config.kd

    return kp, kd

def get_control_torque(q_current, omega, config):
    '''
    Computes the control torque for the configured controller, clipped to
    the actuator limit.

    Args:
        [q_current]: 4-element current orientation quaternion (unitless).
        [omega]: 3-element body-frame angular velocity vector (rad s^-1).
        [config]: Simulation config, providing [controller] (none / pd / contraction).

    Returns:
        3-element array representing the clipped control torque (N m).

    Raises:
        ValueError: If [config.controller] is not a supported controller type.
    '''
    q_error_vec = quaternion_error_vec(q_current, Q_TARGET)

    # 2. Dispatch to the correct controller
    if config.controller == "none":
        tau_c = np.zeros(3)

    elif config.controller == "pd":
        tau_c = pd_controller(q_error_vec, omega, *pd_gains(config), TAU_MAX)

    elif config.controller == "contraction":
        tau_c = contraction_control(q_error_vec, omega, TAU_MAX)

    else:
        raise ValueError(f"Unknown controller type: {config.controller}")
        
    return tau_c


def get_raw_control_torque(q_current, omega, config):
    '''
    Returns the commanded torque before the actuator clip, so callers can tell
    when [get_control_torque] saturated.

    Args:
        [q_current]: 4-element current orientation quaternion (unitless).
        [omega]: 3-element body-frame angular velocity vector (rad s^-1).
        [config]: Simulation config, providing [controller] (none / pd / contraction).

    Returns:
        3-element array representing the unclipped commanded torque (N m).

    Raises:
        ValueError: If [config.controller] is not a supported controller type.
    '''
    q_error_vec = quaternion_error_vec(q_current, Q_TARGET)

    if config.controller == "none":
        return np.zeros(3)

    elif config.controller == "pd":
        return pd_raw(q_error_vec, omega, *pd_gains(config))

    elif config.controller == "contraction":
        return contraction_raw(q_error_vec, omega)

    raise ValueError(f"Unknown controller type: {config.controller}")
