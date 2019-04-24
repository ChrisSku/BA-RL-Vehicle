__all__ = ["Battery"]

INITIAL_SOC = 0.95
INTERNAL_RESISTANCE = 0.1
OPEN_CIRCUIT_VOLTAGE = 340
ACCESSORY_LOAD = 600
ENERGY_CAPACITY = 12.6


class Battery:
    def __init__(self, initial_power):
        self.state_of_charge = INITIAL_SOC * 100
        self.__battery_current = 0
        self.__initial_power = initial_power
        self.empty_battery = False

    def __current_calculation(self, motor_power_input):
        output_power = motor_power_input + ACCESSORY_LOAD
        square_root = (OPEN_CIRCUIT_VOLTAGE ** 2 - (output_power * INTERNAL_RESISTANCE * 4)) ** 0.5
        self.__battery_current = (OPEN_CIRCUIT_VOLTAGE - square_root) / (INTERNAL_RESISTANCE * 2)

    def step(self, motor_power_input):
        self.__current_calculation(motor_power_input)
        self.__initial_power -= self.__battery_current * OPEN_CIRCUIT_VOLTAGE / (
                ENERGY_CAPACITY * 3600 * 1000)
        self.state_of_charge = 100 * self.__initial_power
        if self.state_of_charge <= 5:
            self.empty_battery = True
