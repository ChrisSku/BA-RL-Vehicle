import tensorflow as tf
from oldMethod import first_own as envi
import numpy as np
###############################################
######## PART ONE: NETWORK VARIABLES #########
#############################################

# Observation Space has 4 inputs
num_inputs = 3

num_hidden = 4

# Outputs the probability it should go left
num_outputs = 100

initializer = tf.contrib.layers.variance_scaling_initializer()


###############################################
######## PART TWO: NETWORK LAYERS #########
#############################################

X = tf.placeholder(tf.float32, shape=[None,num_inputs])
hidden_layer_one = tf.layers.dense(X,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)

# Probability to go left
output_layer = tf.layers.dense(hidden_layer_one,num_outputs,activation=tf.nn.sigmoid,kernel_initializer=initializer)

# [ Prob to go left , Prob to go right]
probabilties = tf.concat(axis=1, values=[output_layer, 1 - output_layer])

# Sample 1 randomly based on probabilities
action = tf.multinomial(output_layer, num_samples=1)


init = tf.global_variables_initializer()

###############################################
######## PART THREE: SESSION #########
#############################################

saver = tf.train.Saver()

epi = 500
step_limit = 500
avg_steps = []
env = envi.BEV()
with tf.Session() as sess:
    init.run()
    print("STARTET RUN")
    for i_episode in range(epi):
        print(f'Episode: {i_episode}')
        done = False
        obs = env.reset()
        totalreward = []
        while not done:
            # env.render()
            action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
            obs, reward, done, info = env.step((action_val[0][0]))
            totalreward.append(reward)
        print(f'done Totalreward: {sum(totalreward)}')
        if i_episode % 20 == 0 :
            env.plot()

    obs = env.reset()
    for test in range(step_limit):
        env.render()
        action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
        obs, reward, done, info = env.step(action_val[0][0])
        if done:
            avg_steps.append(test)
            print('Done after {} steps'.format(test))
            break


print("After {} episodes the average cart steps before done was {}".format(epi,np.mean(avg_steps)))
env.close()
