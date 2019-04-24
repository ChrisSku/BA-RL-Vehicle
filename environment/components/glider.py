import numpy as np

__all__ = ["Glider"]

AERO_DRAG_COEFF = 0.38
AIR_DENSITY = 1.23
FRONT_AREA = 2.1
GRAVITY = 9.81
inclination_angle = 0
INERTIAL_MASS_VEH = 2392
MASS_VEH = 2300
ROLLING_RESIST_COEFF = 0.01


class Glider:
    def __init__(self):
        self.velocity = 0
        self.position = 0
        self.tractive_power = 0
        self.acceleration = 0

    def step(self, tractive_force):
        aerodynamic_drag = self.velocity ** 2 * 0.5 * AIR_DENSITY * AERO_DRAG_COEFF * FRONT_AREA * np.sign(
            self.velocity)
        rolling_resistance = MASS_VEH * GRAVITY * ROLLING_RESIST_COEFF * np.sign(self.velocity)
        grade_force = MASS_VEH * GRAVITY * np.sin(inclination_angle)
        inertial_force = tractive_force - aerodynamic_drag - rolling_resistance - grade_force
        self.acceleration = inertial_force / INERTIAL_MASS_VEH
        self.velocity += self.acceleration
        self.position += self.velocity
        self.tractive_power = self.acceleration * tractive_force
