import os
import numpy as np
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def main():
    env = gym.make("imgreg_test-v5")
    data_path = 'data/test/MNIST.h5'
    env.loadData(data_path)
    total_error = 0.0
    act = deepq.load('models/2.pkl')
    episodes = 100
    l1, rl1 = 0.0, 0.0
    for e in range(episodes):
        ob = env.reset()
        done = False
        tstate_history = {}
        while not done:
            # env.render()
            state = ob
            state_string = (str(env.tstate[0]), str(env.tstate[1]))
            if state_string not in tstate_history:
                tstate_history[state_string] = 0
            tstate_history[state_string] += 1
            action = act(np.reshape(state, (1, 2, 64, 64)))
            next_ob, reward, done, _ = env.step(action)
            ob = next_ob
            terminate = any(val >= 2 for key, val in tstate_history.items())
            if terminate == True:
                break
        rl1_ = np.sum(np.abs(env.target))
        l1_ = np.sum(np.abs(env.target - env.tstate))
        rl1 += rl1_
        l1 += l1_
    print("Trained: L1 over {} episodes = {}".format(episodes, l1 / episodes))
    print("Random: L1 over {} episodes = {}".format(episodes, rl1 / episodes))

if __name__ == '__main__':
    main() 

