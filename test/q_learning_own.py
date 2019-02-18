import numpy as np
import Original.first_own as environment

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

Q = np.zeros((len(vel_space), len(acc_space), env.action_space.n))

NUM_ITERATIONS = 300

alpha = 0.8
gamma = 0.5
total = []
for i in range(NUM_ITERATIONS):
    observation = env.reset()
    done = False
    epsilon = 0.6 * (0.99 ** i)
    totalreward = []

    while not done:
        prvState = [vel_space.index(round(observation[0])), acc_space.index(round(observation[1], 2))]
        action = np.argmax(Q[prvState[0], prvState[1], :]) if np.random.random() > epsilon \
            else env.action_space.sample()
        observation, reward, done, index = env.step(action)
        state = [vel_space.index(round(observation[0])), acc_space.index(round(observation[1], 2))]
        Q[prvState[0], prvState[1], action] = (1 - alpha) * Q[prvState[0], prvState[1], action] + alpha * (
                    reward + gamma * np.max(Q[state[0], state[1], :]))
        totalreward.append(reward)
    print(reward)

    plot_done = False
    if i % 10 == 0 and i != 0:
        plot_observation = env.reset()
        while not plot_done:
            plot_prvState = [vel_space.index(round(plot_observation[0])),
                             acc_space.index(round(plot_observation[1], 2))]
            plot_action = np.argmax(Q[plot_prvState[0], plot_prvState[1], :])
            plot_observation, plot_reward, plot_done, plot_index = env.step(plot_action)
            # if i % 100 == 0:
            #     env.render()
        env.plot()
        print(plot_reward)

    if i == NUM_ITERATIONS - 1:
        eval_action = "["
        for t in range(597):
            plot_prvState = [vel_space.index(round(plot_observation[0])),
                             acc_space.index(round(plot_observation[1], 2))]
            plot_action = np.argmax(Q[plot_prvState[0], plot_prvState[1], :])
            plot_observation, plot_reward, plot_done, plot_index = env.step(plot_action)
            eval_action += f"[{t};{plot_action * 100 / (env.action_space.n - 1)}]"
            if i == 596:
                eval_action += "]"
            else:
                eval_action += ","
        env.plot()
    print(f'Episode {i} and totalreward {sum(totalreward)}')

    total.append(sum(totalreward))
    breackit = False
    if i > 10:
        if round(total[-1]) == round(total[-2]) == round(total[-3]):
            breackit = True

    if breackit:
        break
