import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import os
import multiprocessing as mp

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


def heaviside(x, out_x):
    if x < 0:
        return 0
    elif x == 0:
        return out_x
    else:
        return 1


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
        self.propelling_energy = self.total_tractive_energy_j * heaviside(self.total_tractive_energy_j, 0)
        self.braking_energy = self.total_tractive_energy_j * heaviside(-self.total_tractive_energy_j, 0)


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


class BEV:
    def __init__(self):
        if not os.path.exists('figures'):
            os.makedirs('figures')
        date = datetime.datetime.now()
        datestring = date.strftime('%d-%m-%Y_%H-%M')
        self.figuer_path = f'figures-{datestring}'
        if not os.path.exists('figures/' + self.figuer_path):
            os.makedirs('figures/' + self.figuer_path)

        self.max_speed = 250
        self.min_speed = -1
        self.max_acceleration = 5
        self.min_acceleration = -5
        self.max_SOC = 100
        self.min_SOC = 0
        self.plot_num = 0

        self.low = np.array([self.min_acceleration, self.min_SOC])
        self.high = np.array([self.max_acceleration, self.max_SOC])

        self.reset()

    def reset(self, setpoint_vel=30):
        steps_per_sec = 1
        self.driveline = Driveline(steps_per_sec)
        self.glider = Glider(steps_per_sec)
        self.brake = BrakeSystem(steps_per_sec)
        self.battery = Battery(steps_per_sec)
        self.motor = Motor(steps_per_sec)
        self.state = np.array([self.glider.velocity, self.battery.SOC])
        self.action_tracker = []
        self.velocity_traker = []
        self.SOC_tracker = []
        self.position_traker = []
        self.acceleration_tracker = []
        self.setpoint_vel = setpoint_vel
        return 0

    def step(self, action):
        app = action
        bpp = 0
        if self.battery.empty_battery:
            app, bkk = 0, 0
        self.last_position = self.glider.position
        self.brake.step(bpp, self.glider.velocity)
        self.motor.step(app, self.driveline.motor_speed, self.brake.regen_brake_torque)
        self.driveline.step(self.glider.velocity_mps * 3.6, self.motor.motor_net_torque,
                            self.brake.friction_brake_force)
        self.battery.step(self.motor.motor_power_input)
        self.glider.step(self.driveline.net_tractive_force)
        self.velocity_traker.append(self.glider.velocity)
        self.SOC_tracker.append(self.battery.SOC)
        self.position_traker.append(self.glider.position)
        self.acceleration_tracker.append(self.glider.acceleration)
        self.action_tracker.append(action)
        done = self.battery.empty_battery
        reward = -((self.battery.SOC - 100) ** 2) - (heaviside(self.glider.velocity, 0) - 1)*100 + self.glider.position/100
        # if len(self.velocity_traker) > 300:
        #     done = round(sum(self.velocity_traker[-30:])/30) == 0
        # if done:
        #     reward = self.glider.position
        # self.state = (self.glider.acceleration, self.battery.SOC)
        state = np.clip(round(self.glider.velocity_mps * 3.6), 0, 250)
        return reward, state, done, self.glider.position

    def plot(self, epi):
        plt.subplot(5, 1, 1)
        plt.plot(self.velocity_traker)
        plt.ylabel('velocity[kmph]')
        plt.xlabel('time[s]')
        plt.title(f'uhuhh')
        plt.subplot(5, 1, 2)
        plt.plot(self.SOC_tracker)
        plt.ylabel('SOC[%]')
        plt.xlabel('time[s]')
        plt.subplot(5, 1, 3)
        plt.plot(self.position_traker)
        plt.ylabel('Distance[m]')
        plt.xlabel('time[s]')
        plt.subplot(5, 1, 4)
        plt.plot(self.acceleration_tracker)
        plt.ylabel('Acceleration[mpss]')
        plt.xlabel('time[s]')
        plt.subplot(5, 1, 5)
        plt.plot(self.action_tracker)
        plt.ylabel('Pedal[%]')
        plt.xlabel('time[s]')
        plt.savefig(f'figures/{self.figuer_path}/plot-{epi}.png')

        plt.clf()


def run_experiment(env, agent, episode, e_greedy):
    try:
        action = agent.initial_action()
    except AttributeError:
        action = 0
    done = False
    totalreward = []
    while not done:
        reward, next_state, done, position = env.step(action)
        action = agent.step(reward, 0.9, int(next_state), episode, e_greedy)
        totalreward.append(reward)
    return reward, totalreward, position


def epsilon_greedy(q_values, epsilon, episode):
    if epsilon * 0.99 ** episode < np.random.random():
        return np.argmax(q_values)
    else:
        return np.random.randint(np.array(q_values).shape[-1])


def greedy(q_values):
    return np.argmax(q_values)


class GeneralQ(object):

    def __init__(self, number_of_states, number_of_actions, initial_state, target_policy, behaviour_policy, double,
                 step_size=0.1):
        self._q = np.zeros((number_of_states, number_of_actions))
        if double:
            self._q2 = np.zeros((number_of_states, number_of_actions))
        self._s = initial_state
        self._number_of_actions = number_of_actions
        self._step_size = step_size
        self._behaviour_policy = behaviour_policy
        self._target_policy = target_policy
        self._double = double
        self._last_action = 0

    @property
    def q_values(self):
        if self._double:
            return (self._q + self._q2) / 2
        else:
            return self._q

    def step(self, r, g, s, episode, e_greedy):
        # select
        a = self._behaviour_policy(self.q_values[s], episode, e_greedy)
        # update
        if self._double:
            if np.random.random() > 0.5:
                self._q[self._s][self._last_action] += self._step_size * (
                        r + g * np.dot(self._q2[s], self._target_policy(self._q[s], a)) - self._q[self._s][
                    self._last_action])
            else:
                self._q2[self._s][self._last_action] += self._step_size * (
                        r + g * np.dot(self._q[s], self._target_policy(self._q2[s], a)) - self._q2[self._s][
                    self._last_action])
        else:
            self._q[self._s][self._last_action] += self._step_size * (
                    r + g * np.dot(self._q[s], self._target_policy(self._q[s], a)) - self._q[self._s][
                self._last_action])
        self._s = s
        self._last_action = a
        return a


env = BEV()


def target_policy(q, a):
    return np.eye(len(q))[np.argmax(q)]


def behaviour_policy(q, n, e_greedy):
    if e_greedy:
        return epsilon_greedy(q, 0.2, n)
    else:
        return greedy(q)


def finischedLearning(rewards):
    length = len(rewards)
    times = 1 + length / 1000
    if length < 1500:
        return False
    else:
        if np.std(rewards[-20:]) < 10000 * times:
            return True
        else:
            return False


def run_learning():
    agent = GeneralQ(env.max_speed, 101, env.reset(),
                     target_policy, behaviour_policy, double=False)
    rewards = []
    positions = []
    episode = 1
    while not finischedLearning(rewards):
        env.reset()
        reward, totalreward, position = run_experiment(env, agent, (episode - 1), True)
        rewards.append(sum(totalreward))
        positions.append(position)
        print(f'Episode {episode}: Totalreward is {sum(totalreward)} and last reward is {reward}')
        if episode % 100 == 0:
            env.plot(episode)
        episode += 1
    # if (i + 1)  % round(episodes/5) == 0:

    run_experiment(env, agent, episode, False)
    env.plot(episode)
    max_position = max(positions[-20:])
    plt.subplot(1, 1, 1)
    plt.plot(rewards)
    plt.title(f"Set point velocity ")
    plt.ylabel('Totalreward')
    plt.xlabel('Episodes')
    plt.savefig(f'figures/{env.figuer_path}/plot-total.png')
    plt.clf()


if __name__ == '__main__':
    run_learning()
# Run Learning
