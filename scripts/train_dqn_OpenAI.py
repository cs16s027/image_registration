import os
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def play():
    env = gym.make("imgreg_train-v2")
    data_path = 'data/train.h5'
    env.loadData(data_path)
    env.reset()
    env.render()

def main():
    env = gym.make("imgreg_train-v2")
    data_path = 'data/train.h5'
    env.loadData(data_path)
    model = deepq.models.cnn_to_mlp([(16, 8, 4), (32, 4, 2)], [256])
    act = deepq.learn(
        env,
        q_func=model,
        lr = 1e-3,
        max_timesteps = 100000,
        buffer_size = 50000,
        exploration_fraction = 0.3,
        exploration_final_eps = 0.02,
        print_freq = 10,
        gamma = 0.95
    )
    print("Saving model")
    act.save("models/iter_2.pkl")


if __name__ == '__main__':
    main() 

