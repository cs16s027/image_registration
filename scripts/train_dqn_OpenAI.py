import imgreg
import gym

from baselines import deepq

def main():
    env = gym.make("imgreg_train-v2")
    data_path = 'data/train.h5'
    env.loadData(data_path)
    model = deepq.models.cnn_to_mlp([(8, 3, 2), (16, 3, 2), (32, 3, 2)], [512, 256, 64])
    act = deepq.learn(
        env,
        q_func=model,
        lr = 1e-3,
        max_timesteps = 100000,
        buffer_size = 50000,
        exploration_fraction = 0.1,
        exploration_final_eps = 0.02,
        print_freq = 10,
    )
    print("Saving model to imgreg_model.pkl")
    act.save("models/imgreg_model.pkl")


if __name__ == '__main__':
    main() 

