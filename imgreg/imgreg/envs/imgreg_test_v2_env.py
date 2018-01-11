import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import cv2
from copy import deepcopy

class ImgRegTestv2(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.viewer = None
        self.height, self.width = 64, 64
        self.observation_space = spaces.Box(low = 0, high = 63, shape=(self.height, self.width))
        self.action_space = spaces.Discrete(5)
        self.epsilon = 2
        self.reward = 0.0
        self.registered = False

    def _step(self, action):
        reward = self.act(action)
        ob = self._get_obs()
        return ob, reward, self.registered, {'Hi' : 'boss'}

    def _reset(self):
        self.state = np.stack([self.ref_image, self.def_image, self.ref_image], axis = 0)
        self.registered = False
        self.tstate = np.float32([0, 0])
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img, tstate = self._get_obs()
        if mode == 'rgb_array':
            return img      
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _get_obs(self):
        return (self.state, self.tstate)

    def initialize(self, ref_image, def_image):
        self.ref_image = deepcopy(ref_image)
        self.def_image = deepcopy(def_image)
        self.tstate = np.float32([0, 0])
        self.state = np.stack([self.ref_image, self.def_image, self.ref_image], axis = 0)

    def act(self, action):
        if action == 4:
            self.registered = True
            return 0.0
        # Parsing the action
        direction = action / 2
        sign = 1 if action % 2 == 0 else -1
        self.tstate[direction] += sign

        # The action
        self.tmatrix = np.float32([[1, 0, self.tstate[0]], [0, 1, self.tstate[1]]])
        self.trans_image = cv2.warpAffine(self.ref_image, self.tmatrix, (self.height, self.width))
        self.state = np.stack([self.ref_image, self.def_image, self.trans_image], axis = 0)

        # Reward calculation
        return 0.0

ACTION_MEANING = {
        0 : "RIGHT",\
        1 : "LEFT",\
        2 : "UP",\
        3 : "DOWN",\
        4 : "STOP"
}

