import numpy as np

__all__ = ["Motor"]

MOTOR_KC = 0.0452
MOTOR_KI = 0.0167
MOTOR_KW = 5.0664e-05
MOTOR_C = 628.2974
MOTOR_MAX_TRQ_OUT = 500
MOTOR_MAX_PW_OUT = 100000


def motor_losses(motor_net_torque, motor_speed):
    return motor_net_torque ** 2 * MOTOR_KC + \
           np.polyval([MOTOR_KW, 0, MOTOR_KI, MOTOR_C], motor_speed) if motor_speed > 0 else 0


def regen_torque_limiter(regen_brake_torque, allowable_regen_torque):
    return abs(np.clip(regen_brake_torque, allowable_regen_torque, 0))


def motor_torque_limiter(app, motor_speed):
    min_of_max_torque = min(MOTOR_MAX_TRQ_OUT, MOTOR_MAX_PW_OUT / motor_speed)
    max_torque = app / 100 * min_of_max_torque
    allowable_regen_torque = -.5 * min_of_max_torque
    return max_torque, allowable_regen_torque


class Motor:
    def __init__(self):
        self.motor_power_output = 0
        self.motor_power_input = 0
        self.motor_power_losses = 0
        self.motor_net_torque = 0

    def step(self, app, motor_speed, regen_brake_torque):
        max_torque, allowable_regen_torque = motor_torque_limiter(app, motor_speed)
        self.motor_net_torque = max_torque - regen_torque_limiter(regen_brake_torque, allowable_regen_torque)
        self.motor_power_losses = motor_losses(self.motor_net_torque, motor_speed)
        self.motor_power_output = self.motor_net_torque * motor_speed
        self.motor_power_input = self.motor_power_output + self.motor_power_losses
