import numpy as np
import math
import matplotlib.pyplot as plt
import gym
from gym import spaces
import datetime
import os

ACCESSORY_LOAD = 600
AERO_DRAG_COEFF = 0.38
AIR_DENSITY = 1.23
C_REF = 600
ENERGY_CAPACITY = 12.6
FRONT_AREA = 2.1
GRAVITY = 9.81
inclination_angle = 0
INERTIAL_MASS_VEH = 2392
INITIAL_SOC = 0.95
INTERNAL_RESISTANCE = 0.1
K = 1.047162270183853
KC_REF = 0.12
KI_REF = 00.1
KW_REF = 1.2e-05
MASS_VEH = 2300
OPEN_CIRCUIT_VOLTAGE = 340
ROLLING_RESIST_COEFF = 0.01
T_MAX = 500
T_REF = 300
W_MAX = 524
W_REF = 834
WHEEL_RADIUS = 0.34
MOTOR_KC = 0.0452
MOTOR_KI = 0.0167
MOTOR_KW = 5.0664e-05
MOTOR_C = 628.2974
MOTOR_MAX_TRQ_OUT = 500
MOTOR_MAX_PW_OUT = 100000
DRIVELINE_C0 = 600
DRIVELINE_C1 = 0.0120
DRIVELINE_C2 = 4.9e-07
DRIVELINE_G = 3.55
DRIVELINE_SPINNLOSS = 6
DRIVELINE_FTR_MAX = 5000
DRIVER_KP = 350
DRIVER_KI = 0
DRIVER_KD = 0
DRIVER_BRAKE_GAIN = 100
DRIVER_FTR_MAX = 5000
DRIVER_PTR_MAX = 100000
BRAKE_MAX_BRAKE_FORCE = 10000
BRAKE_REGEN_FRACTION = 0.5500
MPS_TO_MPH = 2.23694


class Glider:
    def __init__(self, steps_per_sec=1):
        self.velocity_mps = 0
        self.velocity = 0
        self.position = 0
        self.tractive_power = 0
        self.total_tractive_energy_kwh = 0
        self.total_tractive_energy_j = 0
        self.propelling_energy = 0
        self.braking_energy = 0
        self.steps_per_sec = steps_per_sec
        self.acceleration = 0

    def step(self, tractive_force):
        aerodynamic_drag = self.velocity_mps ** 2 * 0.5 * AIR_DENSITY * AERO_DRAG_COEFF * FRONT_AREA * np.sign(
            self.velocity)
        rolling_resistance = MASS_VEH * GRAVITY * ROLLING_RESIST_COEFF * np.sign(self.velocity)
        grade_force = MASS_VEH * GRAVITY * np.sin(inclination_angle)
        inertial_force = tractive_force - aerodynamic_drag - rolling_resistance - grade_force
        # print(f'\n\nGLider: \ninertial force: {inertial_force}')
        # print(f'aerodynamic: {aerodynamic_drag}')
        # print(f'rolling_resistance: {rolling_resistance}')
        # print(f'gradeforce:{grade_force}')
        self.acceleration = inertial_force / INERTIAL_MASS_VEH
        self.velocity_mps += self.acceleration * 1 / self.steps_per_sec
        # print(f'velocyty: {self.velocity_mps}')
        self.position += self.velocity_mps * 1 / self.steps_per_sec
        self.velocity = self.velocity_mps * MPS_TO_MPH
        self.tractive_power = self.acceleration * tractive_force
        self.total_tractive_energy_j += self.tractive_power * 1 / self.steps_per_sec
        self.total_tractive_energy_kwh = self.total_tractive_energy_j * 0.000278 / 1000
        self.propelling_energy = self.total_tractive_energy_j * np.heaviside(self.total_tractive_energy_j, 0)
        self.braking_energy = self.total_tractive_energy_j * np.heaviside(-self.total_tractive_energy_j, 0)


class Driveline:
    def __init__(self, steps_per_sec=1):
        self.driveline_power_loss = 0
        self.driveline_losses_kWh = 0
        self.driveline_losses_j = 0
        self.driveline_torque_output = 0
        self.motor_speed = 0.00001
        self.net_tractive_force = 0
        self.steps_per_sec = steps_per_sec

    def step(self, vehicle_speed, motor_net_torque, friction_brake_force):
        friction_brake_force = np.clip(friction_brake_force, -10000, 0)
        torque_spin_loss = DRIVELINE_SPINNLOSS if round(vehicle_speed) == 0 else 0
        self.driveline_torque_output = motor_net_torque - torque_spin_loss
        positiv_tractive_force = self.driveline_torque_output * DRIVELINE_G / WHEEL_RADIUS
        self.net_tractive_force = np.clip(positiv_tractive_force + friction_brake_force, -10000, 5000)
        self.motor_speed = np.clip(abs(vehicle_speed / MPS_TO_MPH) * DRIVELINE_G / WHEEL_RADIUS, 0.00001, math.inf)
        self.driveline_power_loss = self.motor_speed * torque_spin_loss
        self.driveline_losses_j += abs(self.driveline_power_loss) * 1 / self.steps_per_sec


class Motor:
    def __init__(self, steps_per_sec=1):
        self.motor_power_output = 0
        self.motor_energy_output = 0
        self.motor_power_input = 0
        self.motor_energy_input = 0
        self.motor_power_losses = 0
        self.motor_energy_losses = 0
        self.motor_net_torque = 0
        self.steps_per_sec = steps_per_sec

    def motor_torque_limiter(self, app, motor_speed):
        min_of_max_torque = min(MOTOR_MAX_TRQ_OUT, MOTOR_MAX_PW_OUT / motor_speed)
        max_torque = app / 100 * min_of_max_torque
        allowable_regen_torque = -.5 * min_of_max_torque
        return max_torque, allowable_regen_torque

    def regen_torque_limiter(self, regen_brake_torque, allowable_regen_torque):
        return abs(np.clip(regen_brake_torque, allowable_regen_torque, 0))

    def motor_losses(self, motor_net_torque, motor_speed):
        return motor_net_torque ** 2 * MOTOR_KC + \
               np.polyval([MOTOR_KW, 0, MOTOR_KI, MOTOR_C], motor_speed) if motor_speed > 0 else 0

    def step(self, app, motor_speed, regen_brake_torque):
        max_torque, allowable_regen_torque = self.motor_torque_limiter(app, motor_speed)
        self.motor_net_torque = max_torque - self.regen_torque_limiter(regen_brake_torque, allowable_regen_torque)
        self.motor_power_losses = self.motor_losses(self.motor_net_torque, motor_speed)
        self.motor_energy_losses += self.motor_power_losses * 1 / self.steps_per_sec
        self.motor_power_output = self.motor_net_torque * motor_speed
        self.motor_energy_output += self.motor_power_output * 1 / self.steps_per_sec
        self.motor_power_input = self.motor_power_output + self.motor_power_losses
        self.motor_energy_input += self.motor_power_input * 1 / self.steps_per_sec


class BrakeSystem:
    def __init__(self, steps_per_sec=1):
        self.steps_per_sec = steps_per_sec
        self.friction_brake_force = 0
        self.regen_brake_torque = 0

    def step(self, bpp, vehicle_speed):
        desired_brake_force = bpp * BRAKE_MAX_BRAKE_FORCE / 100
        self.friction_brake_force = desired_brake_force * (1 - BRAKE_REGEN_FRACTION)
        regen_brake_torque = desired_brake_force * BRAKE_REGEN_FRACTION * WHEEL_RADIUS / DRIVELINE_G
        self.regen_brake_torque = regen_brake_torque if vehicle_speed > 5 else 0


class Battery:
    def __init__(self, steps_per_sec=1):
        self.steps_per_sec = steps_per_sec
        self.SOC = INITIAL_SOC * 100
        self.battery_power_losses = 0
        self.battery_energy_losses = 0
        self.voltage_at_terminals = 0
        self.battery_power_at_terminals = 0
        self.battery_energy_at_terminals = 0
        self.battery_current = 0
        self.initial_power = INITIAL_SOC
        self.empty_battery = False

    def current_calculation(self, motor_power_input):
        # print(f'motor_power_input: {motor_power_input}')
        output_power = motor_power_input + ACCESSORY_LOAD
        squarroot = (OPEN_CIRCUIT_VOLTAGE ** 2 - (output_power * INTERNAL_RESISTANCE * 4)) ** 0.5
        self.battery_current = (OPEN_CIRCUIT_VOLTAGE - squarroot) / (INTERNAL_RESISTANCE * 2)
        # print(f'self.battery_current {self.battery_current}')

    def step(self, motor_power_input):
        self.current_calculation(motor_power_input)
        self.initial_power -= self.battery_current * OPEN_CIRCUIT_VOLTAGE / (
                    ENERGY_CAPACITY * 3600 * 1000) * 1 / self.steps_per_sec
        # print(f'self.initial_power: {self.initial_power}')
        self.SOC = 100 * self.initial_power
        if self.SOC <= 5:
            self.empty_battery = True
        internal_resistance_voltage_drop = self.battery_current * INTERNAL_RESISTANCE
        self.battery_power_losses = self.battery_current * internal_resistance_voltage_drop
        self.battery_energy_losses += self.battery_power_losses * 1 / self.steps_per_sec
        self.voltage_at_terminals = OPEN_CIRCUIT_VOLTAGE - internal_resistance_voltage_drop
        self.battery_power_at_terminals = self.voltage_at_terminals * self.battery_current
        self.battery_energy_at_terminals += self.battery_power_at_terminals * 1 / self.steps_per_sec


class BEV(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, action_space=5):
        if not os.path.exists('figures'):
            os.makedirs('figures')
        date = datetime.datetime.now()
        datestring = date.strftime('%d-%m-%Y_%H-%M')
        self.figuer_path = f'figures-{datestring}'
        if not os.path.exists('figures/' + self.figuer_path):
            os.makedirs('figures/' + self.figuer_path)
        self.max_speed = 150
        self.min_speed = -150
        self.max_acceleration = 5
        self.min_acceleration = -5
        self.max_SOC = 100
        self.min_SOC = 0
        self.plot_num = 0
        self.viewer = None

        self.low = np.array([self.min_speed, self.min_acceleration, self.min_SOC])
        self.high = np.array([self.max_speed, self.max_acceleration, self.max_SOC])
        print(action_space)
        self.action_space = spaces.Discrete(action_space)
        self.observation_space = spaces.Box(self.low, self.high, dtype=float)
        self.reset()

    def reset(self):
        steps_per_sec = 1
        self.driveline = Driveline(steps_per_sec)
        self.glider = Glider(steps_per_sec)
        self.brake = BrakeSystem(steps_per_sec)
        self.battery = Battery(steps_per_sec)
        self.motor = Motor(steps_per_sec)
        self.state = np.array([self.glider.velocity, self.glider.acceleration, self.battery.SOC])
        self.velocity_traker = []
        self.SOC_tracker = []
        self.position_traker = []
        self.acceleration_tracker = []
        self.action = []
        return np.array(self.state)

    def step(self, action_in):

        action = action_in * (100 / (self.action_space.n - 1))

        app, bpp = np.clip(action, 0, 100), np.clip(action, -100, 0)
        if self.battery.empty_battery:
            app, bkk = 0, 0
        self.action.append(action)
        self.brake.step(bpp, self.glider.velocity)
        self.motor.step(app, self.driveline.motor_speed, self.brake.regen_brake_torque)
        self.driveline.step(self.glider.velocity, self.motor.motor_net_torque, self.brake.friction_brake_force)
        self.battery.step(self.motor.motor_power_input)
        self.glider.step(self.driveline.net_tractive_force)
        self.velocity_traker.append(self.glider.velocity)
        self.SOC_tracker.append(self.battery.SOC)
        self.position_traker.append(self.glider.position)
        self.acceleration_tracker.append(self.glider.acceleration)
        done = self.battery.empty_battery
        reward = self.glider.velocity
        if done:
            reward = self.glider.position
        # if len(self.velocity_traker) > 300:
        #     done = round(sum(self.velocity_traker[-30:])/30) == 0
        # if done:
        #     reward = self.glider.position
        self.state = (
        np.clip(self.glider.velocity, self.min_speed, self.max_speed), self.glider.acceleration, self.battery.SOC)
        return np.array(self.state), reward, done, {}

    def _height(self, xs):
        return xs * 0

    def render(self, mode='human'):
        screen_width = 1800
        screen_height = 600
        max_position = 50000
        min_position = -10

        world_width = max_position - min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(min_position, max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

        pos = self.glider.position
        self.cartrans.set_translation((pos - min_position) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(0)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def plot(self, explore=True):
        plt.subplot(3, 2, 1)
        plt.plot(self.velocity_traker)
        plt.ylabel('velocity[mph]')
        plt.xlabel('time[s]')
        plt.subplot(3, 2, 2)
        plt.plot(self.SOC_tracker)
        plt.ylabel('SOC[%]')
        plt.xlabel('time[s]')
        plt.subplot(3, 2, 3)
        plt.plot(self.position_traker)
        plt.ylabel('Distance[m]')
        plt.xlabel('time[s]')
        plt.subplot(3, 2, 4)
        plt.plot(self.acceleration_tracker)
        plt.ylabel('Acceleration[mpss]')
        plt.xlabel('time[s]')
        plt.subplot(3, 2, 5)
        plt.plot(self.action)
        plt.ylabel('Pedal[%]')
        plt.xlabel('time[s]')
        if explore:
            plt.savefig(f'figures/{self.figuer_path}/plot-{self.plot_num}.png')
            self.plot_num += 1
        else:
            plt.savefig(f'figures/{self.figuer_path}/plot-{self.plot_num}-no-explore.png')

        plt.clf()


def runBEV(runtime=100, steps_per_sec=30):
    driveline = Driveline(steps_per_sec)
    glider = Glider(steps_per_sec)
    brake = BrakeSystem(steps_per_sec)
    battery = Battery(steps_per_sec)
    motor = Motor(steps_per_sec)

    bpp, app = 100, 0

    velocity = [0]
    soc = [95]
    position = [0]

    for i in range(runtime * steps_per_sec):
        if i % 2 == 0:
            app = 1
            bpp = 0
        if i + 1 % 2 == 0:
            app = 0
            bpp = 1
        if battery.empty_battery:
            app = 0
            bpp = 0
        brake.step(bpp, glider.velocity)
        motor.step(app, driveline.motor_speed, brake.regen_brake_torque)
        driveline.step(glider.velocity, motor.motor_net_torque, brake.friction_brake_force)
        battery.step(motor.motor_power_input)
        glider.step(driveline.net_tractive_force)
        if i % steps_per_sec == 0 and i != 0:
            velocity.append(glider.velocity)
            soc.append(battery.SOC)
            position.append(glider.position)

        # print('\n\nMOTOR: ')
        # print(f'power input: {motor.motor_power_input}')
        # print(f'energy input: {motor.motor_energy_input}')
        # print(f'net torque: {motor.motor_net_torque}')
        # print(f'power losses: {motor.motor_power_losses}')
        # print(f'energy losses: {motor.motor_energy_losses}')
        # print(f'power output: {motor.motor_power_output}')
        # print(f'energy output: {motor.motor_energy_output}')
        #
        # print('\n\nDRIVELINE:')
        # print(f'net tractive force: {driveline.net_tractive_force}')
        # print(f'motor speed: {driveline.motor_speed}')
        #
        # print('\n\n')
        # print(f'glider.velocity: {glider.velocity}')
        # print(battery.SOC)

    plt.plot(velocity)
    plt.ylabel('velocity[mph]')
    plt.xlabel('time[s]')
    plt.show()
    plt.plot(soc)
    plt.ylabel('SOC[%]')
    plt.xlabel('time[s]')
    plt.show()
    plt.plot(position)
    plt.ylabel('Distance[m]')
    plt.xlabel('time[s]')
    plt.show()
