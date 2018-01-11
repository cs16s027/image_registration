import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import cv2
import h5py
from copy import deepcopy

class ImgRegTrainv2(gym.Env):
    def __init__(self):
        self.height, self.width = 64, 64
        self.observation_space = spaces.Box(low=0, high=63, shape=(3, self.height, self.width))
        self.action_space = spaces.Discrete(5)
        self.epsilon = 2
        self.bonus = 1
        self.registered = False
        self.max_steps = 100

    def _step(self, action):
        self.steps += 1
        reward = self.act(int(action))
        ob = self._get_obs()
        return ob, reward, self.registered, {'Hi' : 'boss'}

    def _reset(self):
        self.initialize()
        self.state = np.stack([self.ref_image, self.def_image, self.ref_image], axis = 0)
        self.registered = False
        self.tstate = np.float32([0, 0])
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def initialize(self):
        self.ref_image = deepcopy(self.X[self.count_in_epoch][0])
        self.def_image = deepcopy(self.X[self.count_in_epoch][1])
        self.target = np.float32(self.Y[self.count_in_epoch])
        self.tstate = np.float32([0, 0])
        self.state = np.stack([self.ref_image, self.def_image, self.ref_image], axis = 0)

        self.count_in_epoch += 1
        if self.count_in_epoch == self.X.shape[0]:
            self.count_in_epoch = 0

    def act(self, action):
        # If the action is stop
        if action == 4 or self.steps == self.max_steps:
            self.registered = True
            D = np.sum((self.tstate - self.target) ** 2)
            if D <= self.epsilon:    
                return self.bonus
            else:
                return 0.0
        else:
            # Get direction of action
            old_tstate = deepcopy(self.tstate)
            direction = int(action / 2)
            sign = 1 if action % 2 == 0 else -1
            self.tstate[direction] += sign

            # The action
            self.tmatrix = np.float32([[1, 0, self.tstate[0]], [0, 1, self.tstate[1]]])
            self.trans_image = cv2.warpAffine(self.ref_image, self.tmatrix, (self.height, self.width))
            self.state = np.stack([self.ref_image, self.def_image, self.trans_image], axis = 0)

            # Immediate rewards
            D_old = np.sum((old_tstate - self.target) ** 2)
            D_new = np.sum((self.tstate - self.target) ** 2)
            distance = float(D_old - D_new)
            if distance < 0.0:
                reward = 0.0
            else:
                reward = 0.0

            return reward

    def loadData(self, data_path):
            dataset = h5py.File(data_path, 'r')
            self.X, self.Y = dataset['X'][:], dataset['Y'][:]
            self.count_in_epoch = 0

ACTION_MEANING = {
        0 : "RIGHT",\
        1 : "LEFT",\
        2 : "DOWN",\
        3 : "UP",\
        4 : "STOP"
}

