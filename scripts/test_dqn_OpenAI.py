import os
import numpy as np
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def main():
    env = gym.make("imgreg_test-v2")
    data_path = 'data/train.h5'
    env.loadData(data_path)
    act = deepq.load('models/iter_1.pkl')
    for e in range(100):
        ob = env.reset()
        done = False
        while not done:
            # env.render()
            state = ob
            action = act(np.reshape(state, (1, 64, 64)))
            next_ob, reward, done, _ = env.step(action)
            ob = next_ob
        print("Target : (%s, %s), Prediction : (%s, %s)" % (env.target[0], env.target[1], env.tstate[0], env.tstate[1]))

if __name__ == '__main__':
    main() 

