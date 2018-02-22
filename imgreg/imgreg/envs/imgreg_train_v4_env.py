import time
import pyglet
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import cv2
import h5py
from copy import deepcopy

ACTION_MEANING = {
        0 : "RIGHT",\
        1 : "LEFT",\
        2 : "DOWN",\
        3 : "UP",\
}

class ImgRegTrainv4(gym.Env):
    def __init__(self):
        self.height, self.width = 256, 256
        self.observation_space = spaces.Box(low = 0, high = 1.0, shape= (2, self.height, self.width))
        self.bound = 25
        self.action_space = spaces.Discrete(4)
        self.registered = False
        self.max_steps = 30
        self.max_steps_max = 30
        self.epochs = 0
        self.steps = 0
        self.track_reward = 0.0

    def _step(self, action):
        self.steps += 1
        reward = self.act(int(action))
        ob = self._get_obs()
        return ob, reward, self.registered, {'Hi' : 'boss'}

    def preprocess(self, state):
        return state / 255.0

    def _reset(self):
        self.initialize()
        self.state = self.preprocess(np.stack([self.def_image, self.trans_image], axis = 0))
        self.registered = False
        self.tstate = np.float32([0, 0])
        self.steps = 0

        return self._get_obs()

    def _get_obs(self):
        return self.state
	
    def _get_image(self):
        return 255 * self.state

    def initialize(self):
        print("Number of steps = {}/{}".format(self.steps, self.max_steps))
        print("Episode-{} in epoch {}, reward = {}".format(self.count_in_epoch, self.epochs, self.track_reward))
        self.track_reward = 0.0
        self.ref_image = deepcopy(self.X[self.count_in_epoch][0])
        self.def_image = deepcopy(self.X[self.count_in_epoch][1])
        self.trans_image = deepcopy(self.ref_image)
        self.target = np.float32(self.Y[self.count_in_epoch])

        self.count_in_epoch += 1

        if self.count_in_epoch == self.X.shape[0]:
            self.count_in_epoch = 0
            self.epochs += 1
            #self.loadData('data/KLA/train/{}.h5'.format(min(self.epochs, 1)))
            x = np.arange(self.X.shape[0])
            np.random.shuffle(x)
            self.X, self.Y = self.X[x], self.Y[x]
            self.max_steps = min(self.max_steps + 5, self.max_steps_max)

    def act(self, action):
        # Get direction of action
        old_tstate = deepcopy(self.tstate)
        direction = int(action / 2)
        sign = 1 if action % 2 == 0 else -1
        update = self.tstate[direction] + sign
        # Check bound 
        if np.abs(update) <= self.bound:
            self.tstate[direction] = update
            # The action
            self.tmatrix = np.float32([[1, 0, self.tstate[0]], [0, 1, self.tstate[1]]])
            self.trans_image = cv2.warpAffine(self.ref_image, self.tmatrix, (self.height, self.width))
            self.state = self.preprocess(np.stack([self.def_image, self.trans_image], axis = 0))

        # Rewards
        D_old = np.abs(old_tstate[direction] - self.target[direction])
        D_new = np.abs(self.tstate[direction] - self.target[direction])
        reward = 1.0 if D_new < D_old else -1.0

        # Additional rewards
        D = np.max(np.abs(self.tstate - self.target))
        if D == 0:
            reward += 5.0
            terminate = True
        else:
            terminate = False

        # Episode termination
        if terminate == True or self.steps == self.max_steps:
            self.registered = True
        
        self.track_reward += reward

        return reward

    def loadData(self, data_path):
        dataset = h5py.File(data_path, 'r')
        self.X, self.Y = dataset['X'][:], dataset['Y'][:]
        self.count_in_epoch = 0
        print("size of the data:", self.X.shape)


