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
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_img = Input(shape = (3, 64, 64), name = 'X')
        x = Conv2D(8, (3, 3), strides = (2, 2), data_format = 'channels_first',\
                        padding = 'same', name = 'conv1')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(16, (3, 3), strides = (2, 2), data_format = 'channels_first',\
                        padding = 'same', name = 'conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), strides = (2, 2), data_format = 'channels_first',\
                        padding = 'same', name = 'conv3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), strides = (2, 2), data_format = 'channels_first',\
                        padding = 'same', name = 'conv4')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten(name = 'flatten')(x)
        x = Dense(512, activation = 'relu', name = 'fc_1')(x)
        x = Dense(256, activation = 'relu', name = 'fc_2')(x)
        x = Dense(128, activation = 'relu', name = 'fc_3')(x)
        y = Dense(5, activation = 'linear', name = 'Y')(x)
        model = Model(input_img, y)

        model.compile(loss='mse',\
                      optimizer=Adam(lr=self.learning_rate))
        print model.summary()
        return model

    # TODO : Break ties arbitrarily
    def act(self, state):
        state = np.reshape(state, (1, 3, 64, 64))
        print self.model.predict(state)[0]
        return np.argmax(self.model.predict(state)[0])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def loadData():
    data = h5py.File('data/test.h5', 'r')
    return data['X'][:], data['Y'][:]

if __name__ == "__main__":
    X, Y = loadData()
    env = gym.make('imgreg_test-v1')
    agent = DQNAgent()
    agent.load('models/imgreg_dqn.h5')

    done = False
    batch_size = 32

    for e in range(EPISODES):
        ref_image, def_image = X[e][0], X[e][1]
        env.initialize(ref_image, def_image)
        ob = env.reset()
        for time in range(50):
            # env.render()
            state, tstate = ob
            action = agent.act(state)
            next_ob, reward, done, _ = env.step(action)
            ob = next_ob
            if done:
                break
        prediction = (next_ob[1][0], next_ob[1][1])
        print "Target : (%s, %s), Prediction : (%s, %s)" % (-Y[e][1], -Y[e][0], prediction[0], prediction[1])
