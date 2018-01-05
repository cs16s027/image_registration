# -*- coding: utf-8 -*-
import os
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

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def loadEnvironment():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.visible_device_list = "1"
    set_session(tf.Session(config=config))

loadEnvironment()

EPISODES = 100

class DQNAgent:
    def __init__(self):
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = deque(maxlen=500)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.action_size = 5

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
                      optimizer=Adam(lr=self.learning_rate, clipnorm = 1.0))
        print model.summary()
        return model

    def actGreedily(self, tstate):
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

    def actExploratorily(self, state):
        state = np.reshape(state, (1, 3, 64, 64))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def remember(self, ob, action, reward, next_ob, done):
        self.memory.append((ob, action, reward, next_ob, done))

    def learn(self):
        states = []
        targets = []
        for ob, action, reward, next_ob, done in self.memory:
        #for ob, action, reward, next_ob, done in minibatch:
            state, tstate = ob
            state = np.reshape(state, (1, 3, 64, 64))
            next_state, next_tstate = next_ob
            next_state = np.reshape(next_state, (1, 3, 64, 64))
            q_target = reward
            if not done:
                #print 'q(s_t+1, a_t+1) =', self.model.predict(next_state)[0][next_action]
                q_target = reward + self.gamma *\
                            np.max(self.model.predict(next_state)[0])
            q_target_f = self.model.predict(state)
            q_target_f[0][action] = q_target
            q_target_f[0] = np.clip(q_target_f[0], -1, 1)
            #print 'target:', q_target_f[0]
            states.append(state)
            targets.append(q_target_f)

        states = np.array(states, dtype = np.float32)
        targets = np.array(targets, dtype = np.float32)
        states = np.reshape(states, (-1, 3, 64, 64))
        targets = np.reshape(targets, (-1, 5))

        # Backpropagate
        self.model.fit(states, targets, epochs = 1, verbose=0, batch_size = self.batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        done = False

        for e in range(EPISODES):
            ref_image, def_image = X[e][0], X[e][1]
            self.target = np.float32([Y[e][0], Y[e][1]])
            env.initialize(ref_image, def_image, self.target)
            ob = env.reset()
            for time in range(500):
                # env.render()
                state, tstate = ob
                if np.random.randn() < 0.5:
                    action = self.actGreedily(tstate)
                else:
                    action = self.actExploratorily(state) 
                next_ob, reward, done, _ = env.step(action)
                self.remember(ob, action, reward, next_ob, done)
                ob = next_ob
                print '#### Step-%s ####' % str(time + 1)
                print 'tstate:', tstate
                print 'action:', action
                print 'reward:', reward
                if done:
                    print self.model.predict(np.reshape(env.state, (1, 3, 64, 64)))[0]
                    print("episode: {}/{}, time: {}".format(e, EPISODES, time))
                    if int(reward) == 1:
                        for _ in range(0):
                            self.remember(ob, action, reward, next_ob, done)
                    break
            if len(self.memory) > 100:
                self.learn()
            if (e + 1) % 10 == 0:
                self.save("models/imgreg_dqn.h5")

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




def loadData():
    data = h5py.File('data/train.h5', 'r')
    return data['X'][:], data['Y'][:]

if __name__ == "__main__":
    X, Y = loadData()
    env = gym.make('imgreg_train-v1')
    agent = DQNAgent()
    agent.batch_size = 16

    epochs = 100

    for epoch in range(epochs):
        print '#### Epoch-%s ####' % epoch
        agent.train()
        os.system('sleep 2')
        os.system('python scripts/test_dqn_v1.py > logs/%s.log' % epoch)

