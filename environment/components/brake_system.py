__all__ = ["BrakeSystem"]

WHEEL_RADIUS = 0.34
DRIVELINE_G = 3.55
BRAKE_MAX_BRAKE_FORCE = 10000
BRAKE_REGEN_FRACTION = 0.5500


class BrakeSystem:
    def __init__(self):
        self.friction_brake_force = 0
        self.regen_brake_torque = 0

    def step(self, bpp, vehicle_speed):
        desired_brake_force = bpp * BRAKE_MAX_BRAKE_FORCE / 100
        self.friction_brake_force = desired_brake_force * (1 - BRAKE_REGEN_FRACTION)
        regen_brake_torque = desired_brake_force * BRAKE_REGEN_FRACTION * WHEEL_RADIUS / DRIVELINE_G
        self.regen_brake_torque = regen_brake_torque if vehicle_speed > 10 else 0

