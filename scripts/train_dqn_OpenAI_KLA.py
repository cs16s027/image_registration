import os
import imgreg
import gym
import tensorflow as tf

from baselines import deepq

def main():
    env = gym.make("imgreg_train-v4")
    data_path = 'data/KLA/train.h5'
    env.loadData(data_path)
    #model = deepq.models.cnn_to_mlp([(16, 16, 4), (16, 8, 2), (16, 4, 2), (16, 4, 2)], [256])
    model = deepq.models.cnn_to_mlp([(8, 3, 2), (8, 3, 2), (8, 3, 2),(8, 3, 2),(8, 3, 2), (8, 3, 2)], [64])
    act = deepq.learn(
        env,
        q_func=model,
        lr = 1e-3,
        max_timesteps = 1000000,
        checkpoint_freq = 1000,
        buffer_size = 50000,
        exploration_fraction = 0.2,
        exploration_final_eps = 0.02,
        print_freq = 10,
        gamma = 0.95,
        batch_size = 64
    )
    print("Saving model")
    act.save("models/KLA/iter_1.2.pkl")

if __name__ == '__main__':
    main() 

