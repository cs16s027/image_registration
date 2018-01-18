import os
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def main():
    env = gym.make("imgreg_train-v3")
    data_path = 'data/train-1.h5'
    env.loadData(data_path)
    model = deepq.models.cnn_to_mlp([(16, 8, 4), (32, 4, 2)], [256])
    act = deepq.learn(
        env,
        q_func=model,
        lr = 1e-3,
        max_timesteps = 1000000,
        checkpoint_freq = 1000,
        buffer_size = 50000,
        exploration_fraction = 0.01,
        exploration_final_eps = 0.02,
        print_freq = 10,
        gamma = 0.95,
        load_model = 'models/iter_5.pkl'
    )
    print("Saving model")
    act.save("models/iter_5.1.pkl")


if __name__ == '__main__':
    main() 

