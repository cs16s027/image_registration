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

class ImgRegTestv2(gym.Env):
    def __init__(self):
        self.viewer = None
        self.height, self.width = 64, 64
        self.observation_space = spaces.Box(low=0, high=63, shape=(2, self.height, self.width))
        self.bound = 25
        self.action_space = spaces.Discrete(4)
        self.registered = False
        self.max_steps = 50
        self.max_steps_min = 50
        self.close = 2
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
        print("Number of steps = {}".format(self.steps))
        print("Episode-{} in epoch {}, max_steps = {}, reward = {}".format(self.count_in_epoch, self.epochs, self.max_steps, self.track_reward))
        self.track_reward = 0.0
        self.ref_image = deepcopy(self.X[self.count_in_epoch][0])
        self.def_image = deepcopy(self.X[self.count_in_epoch][1])
        self.trans_image = deepcopy(self.ref_image)
        self.target = np.float32(self.Y[self.count_in_epoch])

        self.count_in_epoch += 1

        if self.count_in_epoch % 25 == 0:
            if self.max_steps > self.max_steps_min:
                self.max_steps -= 1

        if self.count_in_epoch == self.X.shape[0]:
            self.count_in_epoch = 0
            self.epochs += 1
            x = np.arange(self.X.shape[0])
            np.random.shuffle(x)
            self.X, self.Y = self.X[x], self.Y[x]

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
        reward = 1.0 if D_old - D_new > 0.0 else -1.0
        D = np.max(np.abs(self.tstate - self.target))
        if D == 0.0:
            reward += 5.0

        # Episode termination
        if self.steps == self.max_steps:
            self.registered = True
        
        self.render()
        time.sleep(0.01)
        #print("Action = {}, old = {}, new = {}, reward = {}".format(ACTION_MEANING[action], old_tstate, self.tstate, reward))

        self.track_reward += reward
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
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        def_image, trans_image = arr[0], arr[1]
        image = np.zeros((64, 64))
        image += def_image / 3
        image += trans_image
        if self.window is None:
            height, width = image.shape
            self.window = pyglet.window.Window(width = 5 * width, height = 5 * height, display = self.display)
            self.width = width
            self.height = height
            self.isopen = True
        cv2.imwrite('image.jpg', image)
        image = cv2.imread('image.jpg', 0)
        image = pyglet.image.ImageData(self.width, self.height, 'I', image.tobytes(), pitch = self.width * -1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(2 * self.width, 2 * self.height)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        print('works')
        self.close()

