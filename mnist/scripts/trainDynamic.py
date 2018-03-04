# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque

from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Input, Activation, Flatten
from keras.optimizers import Adam

import gym
import imgreg

import h5py
from copy import deepcopy

class Agent:
    def __init__(self):
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = deque(maxlen = 2000)

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

    def act(self, tstate):
        greedyfellow = np.zeros((4))
        for a in range(4):
            direction = int(a / 2)
            sign = 1 if a % 2 == 0 else -1
            cstate = deepcopy(tstate)
            cstate[direction] += sign
            greedyfellow[a] = np.sum(np.abs(cstate - self.target))
        min_actions = np.where(np.min(greedyfellow) == greedyfellow)[0]
        return min_actions
    
    def remember(self, ob, target):
        self.memory.append((ob, target))

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X = np.array([item[0] for item in minibatch])
        Y = np.array([item[1] for item in minibatch])
        self.model.fit(X, Y, epochs = 1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def loadData(data_paths):
    dataset = h5py.File(data_paths[0], 'r')
    X, Y = dataset['X'][:], dataset['Y'][:]
    for data_path in data_paths[1 : ]:
        dataset = h5py.File(data_path, 'r')
        X, Y = np.concatenate([X, dataset['X'][:]], axis = 0), np.concatenate([Y, dataset['Y'][:]], axis = 0)
    x = np.arange(X.shape[0])
    np.random.shuffle(x)
    X, Y = X[x], Y[x]
    print("size of the data:", X.shape)
    return X, Y

if __name__ == "__main__":
    X, Y = loadData(['data/train/1.h5','data/train/2.h5', 'data/train/3.h5', 'data/train/4.h5', 'data/train/5.h5'])
    env = gym.make('imgreg_train-v5')
    agent = Agent()
    batch_size = 64

    done = False

    EPISODES = X.shape[0]
    for e in range(EPISODES):
        ref_image, def_image = X[e][0], X[e][1]
        agent.target = Y[e]
        env.initialize(ref_image, def_image, agent.target)
        ob = env.reset()
        for time in range(50):
            state, tstate = ob
            actions = agent.act(tstate)
            action = actions[np.random.randint(0, len(actions))]
            target = np.zeros((4))
            if len(actions) == 2:
                target[actions] = [0.5, 0.5]
            else:
                target[actions] = [1.0]
            next_ob, reward, done, _ = env.step(action)
            agent.remember(state, target)
            ob = next_ob
            if done:
                print("episode: {}/{}, time: {}".format(e, EPISODES, time))
                break
        if len(agent.memory) > batch_size:
            agent.learn(batch_size)
        if (e + 1) % 10 == 0:
            agent.save("models/imgreg.h5")

