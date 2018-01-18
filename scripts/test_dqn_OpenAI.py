import os
import numpy as np
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def main():
    env = gym.make("imgreg_test-v2")
    data_path = 'data/test-out.h5'
    env.loadData(data_path)
<<<<<<< HEAD
    #act = deepq.load('models/inprogress.pkl')
    act = deepq.load('models/iter_5.pkl')
    error = 0.0
=======
    act = deepq.load('models/iter_2.pkl')
>>>>>>> 379c626803fc5c353a6d829bafd09fe35717d995
    for e in range(100):
        ob = env.reset()
        done = False
        while not done:
            # env.render()
            state = ob
            action = act(np.reshape(state, (1, 2, 64, 64)))
            next_ob, reward, done, _ = env.step(action)
            ob = next_ob
        print("Target : (%s, %s), Prediction : (%s, %s)" % (env.target[0], env.target[1], env.tstate[0], env.tstate[1]))
        error += np.sum((env.target - env.tstate) ** 2)
    error /= 100
    print("Error = {}".format(error))

if __name__ == '__main__':
    main() 

