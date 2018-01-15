import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import cv2
import h5py
from copy import deepcopy

class ImgRegTestv2(gym.Env):
    def __init__(self):
        self.height, self.width = 64, 64
        self.observation_space = spaces.Box(low=0, high=63, shape=(self.height, self.width))
        self.action_space = spaces.Discrete(4)
        self.registered = False
        self.max_steps = 20
        self.bound = 6
        self.epochs = 1
        self.steps = 0

    def _step(self, action):
        self.steps += 1
        reward = self.act(int(action))
        ob = self._get_obs()
        return ob, reward, self.registered, {'Hi' : 'boss'}

    def _reset(self):
        self.initialize()
        self.state = self.ref_image - self.def_image
        self.registered = False
        self.tstate = np.float32([0, 0])
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def initialize(self):
        print("Number of steps = {}".format(self.steps))
        print("Episode-{} in epoch {}, max_steps = {}".format(self.count_in_epoch, self.epochs, self.max_steps))
        self.ref_image = deepcopy(self.X[self.count_in_epoch][0])
        self.def_image = deepcopy(self.X[self.count_in_epoch][1])
        self.target = np.float32(self.Y[self.count_in_epoch])

        self.count_in_epoch += 1
        if self.count_in_epoch == self.X.shape[0]:
            self.count_in_epoch = 0
            self.epochs += 1

    def act(self, action):
        # Get direction of action
        old_tstate = deepcopy(self.tstate)
        direction = int(action / 2)
        sign = 1 if action % 2 == 0 else -1
        update = self.tstate[direction] + sign
        # Check bound 
        if update <= self.bound:
            self.tstate[direction] = update
            # The action
            self.tmatrix = np.float32([[1, 0, self.tstate[0]], [0, 1, self.tstate[1]]])
            self.trans_image = cv2.warpAffine(self.ref_image, self.tmatrix, (self.height, self.width))
            self.state = self.trans_image - self.def_image

        if self.steps == self.max_steps:
            self.registered = True
        return 0.0


    def loadData(self, data_path):
        dataset = h5py.File(data_path, 'r')
        self.X, self.Y = dataset['X'][:], dataset['Y'][:]
        self.count_in_epoch = 0
        print("size of the data:", self.X.shape)

ACTION_MEANING = {
        0 : "RIGHT",\
        1 : "LEFT",\
        2 : "DOWN",\
        3 : "UP",\
}

