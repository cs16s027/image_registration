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

class ImgRegTestv4(gym.Env):
    def __init__(self):
        self.viewer = None
        self.height, self.width = 256, 256
        self.observation_space = spaces.Box(low = 0, high = 1.0, shape= (2, self.height, self.width))
        self.bound = 25
        self.action_space = spaces.Discrete(4)
        self.registered = False
        self.max_steps = 20
        self.epochs = 1
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

        # Episode termination
        if self.steps == self.max_steps:
            self.registered = True
        
        self.track_reward += reward

        #self.render()
        #time.sleep(0.001)

        return reward

    def loadData(self, data_path):
        dataset = h5py.File(data_path, 'r')
        self.X, self.Y = dataset['X'][:], dataset['Y'][:]
        self.count_in_epoch = 0
        print("size of the data:", self.X.shape)

    def render(self, mode = 'human', close = False):
        img = self._get_image()
        if self.viewer is None:
            self.viewer = SimpleImageViewer()
        self.viewer.imshow(img)

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.write_to_disk = 0
        self.switch = True
    def imshow(self, arr):
        def_image, trans_image = arr[0], arr[1]
        image = np.zeros((256, 256))
        if self.write_to_disk %10 == 0:
            self.switch = False
        elif self.write_to_disk % 5 == 0:
            self.switch = True
        if self.switch == True:
            image += def_image / 3
        else:
            image += trans_image
        # Begin video processing
        cv2.imwrite('videos/KLA/1.1/{:06d}.jpg'.format(self.write_to_disk), image)
        self.write_to_disk += 1
        # End video processing
        cv2.imwrite('image.jpg', image)
        image = cv2.imread('image.jpg', 0)

