# from .push_controller import get_push_control
from .miscalibrated_push_controller import get_push_control

from gym.envs.robotics import FetchPushEnv

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import time


class MyFetchPushEnv(gym.Env):
    """
    Just to override the broken rendering.
    """

    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make("FetchPush-v1")
        self.metadata = self.fetch_env.metadata
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space

    def step(self, action):
        return self.fetch_env.step(action)

    def reset(self):
        return self.fetch_env.reset()

    def seed(self, seed=0):
        return self.fetch_env.seed(seed=seed)

    def render(self, mode="human", *args, **kwargs):
        # See https://github.com/openai/gym/issues/1081
        self.fetch_env.env._render_callback()
        if mode == 'rgb_array':
            self.fetch_env.env._get_viewer().render()
            width, height = 3350, 1800
            data = self.fetch_env.env._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self.fetch_env.env._get_viewer().render()

        return self.fetch_env.render(*args, **kwargs)


class ResidualFetchPushEnv(gym.Env):

    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make("FetchPush-v1")

        for i in range(len(self.fetch_env.env.sim.model.geom_friction)):
            self.fetch_env.env.sim.model.geom_friction[i] = [18e-2, 5.e-3, 1e-4] # [1.e+00 5.e-03 1.e-04]

        self.metadata = self.fetch_env.metadata
        self.hardcoded_controller = None
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space #spaces.Box(low=-np.inf, high=np.inf, shape=(28,)) #self.fetch_env.observation_space

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_push_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = self.fetch_env.step(action)
        self._last_observation = observation
        
        return observation, reward, done, debug_info

    def reset(self):
        observation = self.fetch_env.reset()
        self._last_observation = observation
        return observation

    def seed(self, seed=0):
        return self.fetch_env.seed(seed=seed)

    def render(self, mode="human", *args, **kwargs):
        # See https://github.com/openai/gym/issues/1081
        self.fetch_env.env._render_callback()
        if mode == 'rgb_array':
            self.fetch_env.env._get_viewer().render()
            width, height = 3350, 1800
            data = self.fetch_env.env._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self.fetch_env.env._get_viewer().render()

        return self.fetch_env.render(*args, **kwargs)

    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

class FetchPushHighFrictionEnv(gym.Env):

    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make("FetchPush-v1")

        for i in range(len(self.fetch_env.env.sim.model.geom_friction)):
            self.fetch_env.env.sim.model.geom_friction[i] = [18e-2, 5.e-3, 1e-4] # [1.e+00 5.e-03 1.e-04]

        self.metadata = self.fetch_env.metadata

        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space 

    def step(self, action):
        return self.fetch_env.step(action)

    def reset(self):
        return self.fetch_env.reset()

    def seed(self, seed=0):
        return self.fetch_env.seed(seed=seed)

    def render(self, mode="human", *args, **kwargs):
        # See https://github.com/openai/gym/issues/1081
        self.fetch_env.env._render_callback()
        if mode == 'rgb_array':
            self.fetch_env.env._get_viewer().render()
            width, height = 3350, 1800
            data = self.fetch_env.env._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self.fetch_env.env._get_viewer().render()

        return self.fetch_env.render(*args, **kwargs)

    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

