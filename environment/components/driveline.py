import numpy as np
import math

__all__ = ["Driveline"]

WHEEL_RADIUS = 0.34
DRIVELINE_G = 3.55
DRIVELINE_SPINNLOSS = 6


class Driveline:
    def __init__(self):
        self.driveline_torque_output = 0
        self.motor_speed = 0.00001
        self.net_tractive_force = 0

    def step(self, vehicle_speed, motor_net_torque, friction_brake_force):
        friction_brake_force = np.clip(friction_brake_force, -10000, 0)
        torque_spin_loss = 0 if round(vehicle_speed) == 0 else DRIVELINE_SPINNLOSS
        self.driveline_torque_output = motor_net_torque - torque_spin_loss
        positiv_tractive_force = self.driveline_torque_output * DRIVELINE_G / WHEEL_RADIUS
        self.net_tractive_force = np.clip(positiv_tractive_force + friction_brake_force, -10000, 5000)
        self.motor_speed = np.clip(abs(vehicle_speed) * DRIVELINE_G / WHEEL_RADIUS, 0.00001, math.inf)
