import numpy as np

from z_main.config import CONTROLLER_TYPE, KP, KD, TAU_MAX, Q_TARGET

from control.pd import pd_controller
from control.contraction import contraction_control

from tumble.quaternion import quaternion_error_vec

def get_control_torque(q_current, omega):
    q_error_vec = quaternion_error_vec(q_current, Q_TARGET)
    
    # 2. Dispatch to the correct controller
    if CONTROLLER_TYPE == "pd":
        tau_c = pd_controller(q_error_vec, omega, KP, KD, TAU_MAX)
        
    elif CONTROLLER_TYPE == "contraction":
        tau_c = contraction_control(q_error_vec, omega, TAU_MAX)
        
    else:
        raise ValueError(f"Unknown controller type: {CONTROLLER_TYPE}")
        
    return tau_c
