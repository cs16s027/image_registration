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
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = deque(maxlen=2000)

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
        y = Dense(4, activation = 'linear', name = 'Y')(x)
        model = Model(input_img, y)

        model.compile(loss='mse',\
                      optimizer=Adam(lr=self.learning_rate))
        print model.summary()
        return model

    def act(self, tstate):
        greedyfellow = []
        for a in range(4):
            cstate = deepcopy(tstate)
            direction = a / 2
            sign = 1 if a % 2 == 0 else -1
            cstate[direction] += sign
            greedyfellow.append(np.sum((cstate - self.target) ** 2))
        greedyfellow = np.array(greedyfellow)
        min_actions = np.argwhere(greedyfellow == np.min(greedyfellow))[0]
        index = np.random.randint(0, len(min_actions))
        return min_actions[index]
    
    def remember(self, ob, action, reward, next_ob, done):
        self.memory.append((ob, action, reward, next_ob, done))

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for ob, action, reward, next_ob, done in minibatch:
            state, tstate = ob
            state = np.reshape(state, (1, 3, 64, 64))
            next_state, next_tstate = next_ob
            next_state = np.reshape(next_state, (1, 3, 64, 64))
            q_target = reward
            if not done:
                next_action = self.act(next_tstate)
                #print 'q(s_t+1, a_t+1) =', self.model.predict(next_state)[0][next_action]
                q_target = reward + self.gamma *\
                            self.model.predict(next_state)[0][next_action]
            q_target_f = self.model.predict(state)
            q_target_f[0][action] = q_target
            #print 'target:', q_target_f[0]
            self.model.fit(state, q_target_f, epochs = 1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def loadData():
    data = h5py.File('data/train.h5', 'r')
    return data['X'][:], data['Y'][:]

if __name__ == "__main__":
    X, Y = loadData()
    env = gym.make('imgreg_train-v0')
    agent = DQNAgent()
    batch_size = 64

    done = False

    for e in range(EPISODES):
        ref_image, def_image = X[e][0], X[e][1]
        agent.target = np.float32([Y[e][0], Y[e][1]])
        env.initialize(ref_image, def_image, agent.target)
        ob = env.reset()
        for time in range(500):
            # env.render()
            state, tstate = ob
            action = agent.act(tstate)
            next_ob, reward, done, _ = env.step(action)
            agent.remember(ob, action, reward, next_ob, done)
            ob = next_ob
            #print '#### Step-%s ####' % str(time + 1)
            #print 'tstate:', tstate
            #print 'action:', action
            #print 'reward:', reward
            if done:
                print("episode: {}/{}, time: {}".format(e, EPISODES, time))
                break
        if len(agent.memory) > batch_size:
            agent.learn(batch_size)
        if (e + 1) % 10 == 0:
            agent.save("models/imgreg_dqn.h5")
