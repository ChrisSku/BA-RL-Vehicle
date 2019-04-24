import agent
import environment
import numpy as np
import matplotlib.pyplot as plt
import sys

def finished_learning(rewards):
    length = len(rewards)
    times = 1 + length / 1000
    if length < 150:
        return False
    else:
        if np.std(rewards[-20:]) < 10000 * times:
            return True
        else:
            return False


def run_experiment(env, ag):
    done = False
    totalreward = []
    while not done:
        action = ag.get_action()
        reward, next_state, done, position = env.step(action)
        ag.step(reward, int(next_state))
        totalreward.append(reward)

    return reward, totalreward, position




class Learning:
    def __init__(self, set_point_vel):
        self.environment = environment.BEV(set_point_vel)
        self.agent = agent.QLearning(204, 101, self.environment.reset(), 0.2, 0.9, 0.1)
        self.results = []

    def start(self):
        rewards = []
        positions = []
        episode = 1
        while not finished_learning(rewards):
            self.environment.reset()
            reward, totalreward, position = run_experiment(self.environment, self.agent)
            rewards.append(sum(totalreward))
            positions.append(position)
            if episode % 50 == 0:
                print(f'Episode {episode}: Totalreward is {sum(totalreward)} and last reward is {reward}')
            if episode % 100 == 0:
                self.environment.plot(episode)
            episode += 1
        # if (i + 1)  % round(episodes/5) == 0:
        # epsilon = self.agent.epsilon
        # self.agent.epsilon = 0
        run_experiment(self.environment, self.agent)
        # self.agent.epsilon = epsilon
        self.environment.plot(episode/100)
        plt.subplot(1, 1, 1)
        plt.plot(rewards)
        plt.title(f"Set point velocity ")
        plt.ylabel('Totalreward')
        plt.xlabel('Episodes')
        plt.savefig(f'figures/{self.environment.path}/plot-total.png')
        plt.clf()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("Start Learning with speed: " + sys.argv[1])
        learning = Learning(120)
        learning.start()
    else:
        for i in range(200, 0, -1):
            print(f'Start Learning with speed: {i}')
            learning = Learning(i)
            learning.start()
