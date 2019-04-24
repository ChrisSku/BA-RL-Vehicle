import numpy as np


class QLearning(object):

    def __init__(self, number_of_states, number_of_actions, initial_state, epsilon, gamma, step_size):
        self._q = np.zeros((number_of_states, number_of_actions))
        self._s = initial_state
        self._a = 0
        self._number_of_actions = number_of_actions
        self.epsilon = epsilon
        self._gamma = gamma
        self._step_size = step_size

    def get_action(self):
        if self.epsilon > np.random.random():
            self._a = round((self._number_of_actions - 1) * np.random.random())
        else:
            self._a = np.argmax(self._q[self._s])
        return self._a

    def step(self, r, s):
        self._q[self._s][self._a] += self._step_size * \
                                     (r + self._gamma * np.max(self._q[s]) - self._q[self._s][self._a])
        self._s = s
