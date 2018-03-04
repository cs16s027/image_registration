# -*- coding: utf-8 -*-
import random
import gym
import imgreg
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Input, Activation, Flatten
from keras.optimizers import Adam
from copy import deepcopy
import h5py

EPISODES = 100

class DQNAgent:
    def __init__(self):
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_img = Input(shape = (2, 64, 64), name = 'X')
        x = Conv2D(8, (8, 8), strides = (4, 4), data_format = 'channels_first',\
                        padding = 'same', name = 'conv1')(input_img)
        x = Activation('relu')(x)
        x = Conv2D(16, (4, 4), strides = (2, 2), data_format = 'channels_first',\
                        padding = 'same', name = 'conv2')(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), strides = (1, 1), data_format = 'channels_first',\
                        padding = 'same', name = 'conv3')(x)
        x = Activation('relu')(x)
        x = Flatten(name = 'flatten')(x)
        x = Dense(256, activation = 'relu', name = 'fc')(x)
        y = Dense(4, activation = 'softmax', name = 'Y')(x)
        model = Model(input_img, y)

        model.compile(loss = 'categorical_crossentropy',\
                      optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    # TODO : Break ties arbitrarily
    def act(self, state):
        state = np.reshape(state, (1, 2, 64, 64))
        return np.argmax(self.model.predict(state)[0])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def loadData():
    data = h5py.File('data/test/MNIST.h5', 'r')
    return data['X'][:], data['Y'][:]

if __name__ == "__main__":
    X, Y = loadData()
    env = gym.make('imgreg_test-v5')
    agent = DQNAgent()
    agent.load('models/imgreg.h5')

    EPISODES = min(EPISODES, X.shape[0])
    done = False
    batch_size = 32

    l1, rl1 = 0.0, 0.0
    for e in range(EPISODES):
        ref_image, def_image = X[e][0], X[e][1]
        target = Y[e]
        env.initialize(ref_image, def_image)
        ob = env.reset()
        done = False
        tstate_history = {}
        terminate = False
        while not done:
            # env.render()
            state_string = (str(env.tstate[0]), str(env.tstate[1]))
            if state_string not in tstate_history:
                tstate_history[state_string] = 0
            tstate_history[state_string] += 1
            state, tstate = ob
            action = agent.act(state)
            next_ob, reward, done, _ = env.step(action)
            ob = next_ob
            terminate = any(val >= 2 for key, val in tstate_history.items())
            if terminate == True:
                break
        rl1_ = np.sum(np.abs(target))
        l1_ = np.sum(np.abs(target - env.tstate))
        rl1 += rl1_
        l1 += l1_
    print("Trained: L1 over {} episodes = {}".format(EPISODES, l1 / EPISODES))
    print("Random: L1 over {} episodes = {}".format(EPISODES, rl1 / EPISODES))

