import numpy as np

from z_main.config import KP, KD, TAU_MAX, Q_TARGET

from control.pd import pd_controller, pd_raw
from control.contraction import contraction_control, contraction_raw

from tumble.quaternion import quaternion_error_vec

def pd_gains(config):
    '''
    Returns the PD gains ([Kp], [Kd]), taking the config overrides when set.
    '''
    kp = KP if config.kp is None else config.kp
    kd = KD if config.kd is None else config.kd

    return kp, kd

def get_control_torque(q_current, omega, config):
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
    '''
    q_error_vec = quaternion_error_vec(q_current, Q_TARGET)

    if config.controller == "none":
        return np.zeros(3)

    elif config.controller == "pd":
        return pd_raw(q_error_vec, omega, *pd_gains(config))

    elif config.controller == "contraction":
        return contraction_raw(q_error_vec, omega)

    raise ValueError(f"Unknown controller type: {config.controller}")
