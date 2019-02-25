import numpy as np
import random as rnd
from oldMethod import first_own as environment

env = environment.BEV(action_space=21)

step_size = 200

acc_space = []

i = env.observation_space.low[1]

while not i > env.observation_space.high[1]:
    acc_space.append(round(i, 2))
    i = round(i + 0.01, 2)

vel_space = []

j = env.observation_space.low[0]

while not j > env.observation_space.high[0]:
    vel_space.append(round(j))
    j = round(j + 1)

Q1 = np.zeros((len(vel_space), len(acc_space), env.action_space.n))
Q2 = np.zeros((len(vel_space), len(acc_space), env.action_space.n))
NUM_ITERATIONS = 1000

alpha = 0.8
gamma = 0.8
total = []
miles = []

for i in range(NUM_ITERATIONS):
    observation = env.reset()
    done = False
    epsilon = 0.2
    totalreward = []
    while not done:
        prvState = [vel_space.index(round(observation[0])), acc_space.index(round(observation[1], 2))]
        Q = np.add(Q1[prvState[0], prvState[1], :], Q2[prvState[0], prvState[1], :])
        action = np.argmax(Q) if np.random.random() > epsilon \
            else env.action_space.sample()
        observation, reward, done, position = env.step(action)
        state = [vel_space.index(round(observation[0])), acc_space.index(round(observation[1], 2))]
        if bool(rnd.getrandbits(1)):
            action_star = np.argmax(Q1[prvState[0], prvState[1], :])
            Q1[prvState[0], prvState[1], action] = (1 - alpha) * Q1[prvState[0], prvState[1], action] + alpha * (
                    reward + Q2[state[0], state[1], action_star])
        else:
            action_star = np.argmax(Q2[prvState[0], prvState[1], :])
            Q2[prvState[0], prvState[1], action] = (1 - alpha) * Q2[prvState[0], prvState[1], action] + alpha * (
                    reward + Q1[state[0], state[1], action_star])
        totalreward.append(reward)
    miles.append(position)
    if reward > 80000:
        env.plot()
    # print(reward)
    if i == NUM_ITERATIONS - 1:
        env.plot()
    # plot_done = False
    # if i % 10 == 0 and i != 0:
    #     plot_observation = env.reset()
    #     while not plot_done:
    #         plot_prvState = [vel_space.index(round(plot_observation[0])),
    #                          acc_space.index(round(plot_observation[1], 2))]
    #         plot_action = np.argmax(Q[plot_prvState[0], plot_prvState[1], :])
    #         plot_observation, plot_reward, plot_done, plot_index = env.step(plot_action)
    #         # if i % 100 == 0:
    #         #     env.render()
    #     env.plot()
    #     print(plot_reward)
    #
    # if i == NUM_ITERATIONS - 1:
    #     eval_action = "["
    #     for t in range(597):
    #         plot_prvState = [vel_space.index(round(plot_observation[0])),
    #                          acc_space.index(round(plot_observation[1], 2))]
    #         plot_action = np.argmax(Q[plot_prvState[0], plot_prvState[1], :])
    #         plot_observation, plot_reward, plot_done, plot_index = env.step(plot_action)
    #         eval_action += f"[{t};{plot_action * 100 / (env.action_space.n - 1)}]"
    #         if i == 596:
    #             eval_action += "]"
    #         else:
    #             eval_action += ","
    #     env.plot()
    print(f'Episode {i} and totalreward {sum(totalreward)} and position {position}')

    total.append(sum(totalreward))
    if i % 100 == 0 and i != 0:
        env.plotTotalreward(miles)

    breackit = False
    if i > 10:
        if round(total[-1], 3) == round(total[-2], 3) == round(total[-3], 3):
            breackit = True

    if breackit:
        break

observation = env.reset()
done = False
totalreward = []
while not done:
    prvState = [vel_space.index(round(observation[0])), acc_space.index(round(observation[1], 2))]
    Q = np.add(Q1[prvState[0], prvState[1], :], Q2[prvState[0], prvState[1], :])
    action = np.argmax(Q)
    observation, reward, done, index = env.step(action)
env.plot(False)
env.plotTotalreward(miles)
