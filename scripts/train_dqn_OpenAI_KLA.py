import os
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def main():
    env = gym.make("imgreg_train-v4")
    data_path = 'data/KLA/train/3.h5'
    env.loadData(data_path)
    model = deepq.models.cnn_to_mlp([(16, 8, 4), (32, 8, 4), (64, 4, 2), (32, 3, 1)], [256])
    act = deepq.learn(
        env,
        q_func=model,
        lr = 1e-3,
        max_timesteps = 50000,
        checkpoint_freq = 1000,
        buffer_size = 50000,
        exploration_fraction = 0.3,
        exploration_final_eps = 0.02,
        print_freq = 10,
        gamma = 0.95,
        batch_size = 32,
        load_model = 'models/KLA/2.pkl'
    )
    print("Saving model")
    act.save("models/KLA/3.pkl")

if __name__ == '__main__':
    main() 

