import numpy as np
import matplotlib.pyplot as plt
from environment.components import Motor, Battery, Glider, Driveline, BrakeSystem
import datetime
import os

__all__ = ['BEV']


def get_figure_path():
    if not os.path.exists('figures'):
        os.makedirs('figures')
    date = datetime.datetime.now()
    datestring = date.strftime('%d-%m-%Y_%H-%M')
    return f'figures-{datestring}'


def plot_time(array, name, einheit, episode, path):
    plot_path = f'figures/{path}/{name}'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plt.plot(array)
    plt.ylabel(f'{name}[{einheit}]')
    plt.xlabel('time[s]')
    plt.title(f'{name}-time')
    plt.savefig(f'{plot_path}/plot-{name}-{episode}.png')
    plt.clf()


# !!!! Belohnung anhand von reichweite pro SOC und diese Maximieren !!!
class BEV(object):
    def __init__(self, set_point_velocity):
        self.path = get_figure_path()
        if not os.path.exists('figures/' + get_figure_path()):
            os.makedirs('figures/' + get_figure_path())
        self.set_point_velocity = set_point_velocity
        self.reset()

    def reset(self):
        self.driveline = Driveline()
        self.glider = Glider()
        self.brake = BrakeSystem()
        self.battery = Battery(1)
        self.motor = Motor()
        self.initial_state = np.array([self.glider.velocity, self.battery.state_of_charge])
        self.action_tracker = []
        self.velocity_traker = []
        self.SOC_tracker = []
        self.position_traker = []
        self.acceleration_tracker = []
        return np.clip(round(self.glider.velocity * 3.6), 0, 203)

    def __component_step(self, action):
        app = action
        bpp = 0
        if self.battery.empty_battery:
            app, bkk = 0, 0
        self.brake.step(bpp, self.glider.velocity)
        self.motor.step(app, self.driveline.motor_speed, self.brake.regen_brake_torque)
        self.driveline.step(self.glider.velocity, self.motor.motor_net_torque,
                            self.brake.friction_brake_force)
        self.battery.step(self.motor.motor_power_input)
        self.glider.step(self.driveline.net_tractive_force)

        self.velocity_traker.append(self.glider.velocity * 3.6)
        self.SOC_tracker.append(self.battery.state_of_charge)
        self.position_traker.append(self.glider.position)
        self.acceleration_tracker.append(self.glider.acceleration)
        self.action_tracker.append(action)

    def step(self, action):
        position = self.glider.position
        soc = self.battery.state_of_charge
        self.__component_step(action)
        done = self.battery.empty_battery
        reward = ( self.glider.position - position) / (soc - self.battery.state_of_charge)
        # reward = -((self.glider.velocity * 3.6 - self.set_point_velocity)**2)
        state = np.clip(round(self.glider.velocity * 3.6), 0, 203)
        return reward, state, done, self.glider.position

    def plot(self, episode):

        plot_time(self.velocity_traker, "Velocity", "km/h", episode, self.path)
        plot_time(self.SOC_tracker, "SOC", "%", episode, self.path)
        plot_time(self.position_traker, "Distance", "m", episode, self.path)
        plot_time(self.action_tracker, "Pedal", "%", episode, self.path)
        plot_time(self.acceleration_tracker, "Acceleration", "m/s^2", episode, self.path)
