import os
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def main():
    env = gym.make("imgreg_train-v5")
    data_paths = ['data/train/1.h5', 'data/train/2.h5']
    env.loadData(data_paths)
    model = deepq.models.cnn_to_mlp([(16, 8, 4), (32, 4, 2), (32, 3, 1)], [256])
    act = deepq.learn(
        env,
        q_func=model,
        lr = 1e-3,
        max_timesteps = 100000,
        checkpoint_freq = 1000,
        buffer_size = 10000,
        exploration_fraction = 0.3,
        exploration_final_eps = 0.02,
        print_freq = 10,
        gamma = 0.95,
        batch_size = 64,
        load_model = None
    )
    print("Saving model")
    act.save("models/2.1.pkl")

if __name__ == '__main__':
    main() 

