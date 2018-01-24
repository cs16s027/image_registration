import os
import numpy as np
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def main():
    env = gym.make("imgreg_test-v2")
    data_path = 'data/test.h5'
    env.loadData(data_path)
    total_error = 0.0
    act = deepq.load('models/iter_5.pkl')
    episodes = env.X.shape[0]
    episodes = 1000
    l1, rl1 = 0.0, 0.0
    l2, rl2 = 0.0, 0.0
    linf, rlinf = 0.0, 0.0
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
            terminate = any(val >= 3 for key, val in tstate_history.items())
            if terminate == True:
                break
        rl1_ = np.sum(np.abs(env.target))
        rl2_ = np.sqrt(np.sum((env.target) ** 2) )
        rlinf_ = np.max(np.abs(env.target))
        l1_ = np.sum(np.abs(env.target - env.tstate))
        l2_ = np.sqrt(np.sum((env.target - env.tstate) ** 2) )
        linf_ = np.max(np.abs(env.target - env.tstate))
        print("L1 = {}, L2 = {}, Linf = {}".format(l1_, l2_, linf_))
        rl1 += rl1_
        rl2 += rl2_
        rlinf += rlinf_
        l1 += l1_
        l2 += l2_
        linf += linf_
    print("Trained: L1 over {} episodes = {}".format(episodes, l1 / episodes))
    print("Trained: L2 over {} episodes = {}".format(episodes, l2 / episodes))
    print("Trained: Linf over {} episodes = {}".format(episodes, linf / episodes))

    print("Random: L1 over {} episodes = {}".format(episodes, rl1 / episodes))
    print("Random: L2 over {} episodes = {}".format(episodes, rl2 / episodes))
    print("Random: Linf over {} episodes = {}".format(episodes, rlinf / episodes))

if __name__ == '__main__':
    main() 

