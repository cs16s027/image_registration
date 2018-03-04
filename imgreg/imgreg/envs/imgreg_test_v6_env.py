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
        4 : "STOP"
}

class ImgRegTestv6(gym.Env):
    def __init__(self):
        self.viewer = None
        self.height, self.width = 64, 64
        self.observation_space = spaces.Box(low = 0, high = 1.0, shape= (2, self.height, self.width))
        self.bound = 25
        self.action_space = spaces.Discrete(5)
        self.registered = False
        self.max_steps = 50
        self.steps = 0

    def _step(self, action):
        self.steps += 1
        reward = self.act(int(action))
        ob = self._get_obs()
        return ob, reward, self.registered, {'Hi' : 'boss'}

    def preprocess(self, state):
        return state / 255.0

    def _reset(self):
        self.state = self.preprocess(np.stack([self.def_image, self.trans_image], axis = 0))
        self.registered = False
        self.tstate = np.float32([0, 0])
        self.steps = 0

        return self._get_obs()

    def _get_obs(self):
        return self.state, self.tstate
	
    def _get_image(self):
        return 255 * self.state

    def initialize(self, ref_image, def_image):
        self.ref_image = ref_image
        self.def_image = def_image
        self.trans_image = ref_image

    def act(self, action):
        if action != 4:
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

        if action == 4 or self.steps == self.max_steps:
            self.registered = True
        
        self.render()
        return 0.0

    def render(self, mode = 'human', close = False):
        img = self._get_image()
        if self.viewer is None:
            self.viewer = SimpleImageViewer()
        self.viewer.imshow(img)

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.write_to_disk = 0
    def imshow(self, arr):
        def_image, trans_image = arr[0], arr[1]
        image = np.zeros((64, 64))
        image += def_image / 3
        image += trans_image
        # Begin video processing
        cv2.imwrite('videos/{:06d}.jpg'.format(self.write_to_disk), image)
        self.write_to_disk += 1
        # End video processing
        cv2.imwrite('image.jpg', image)
        image = cv2.imread('image.jpg', 0)

