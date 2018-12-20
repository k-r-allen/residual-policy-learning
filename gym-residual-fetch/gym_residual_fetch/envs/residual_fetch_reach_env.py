from .reaching_controller import create_reach_controller_from_env

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import time



class ResidualFetchReachEnv(gym.Env):

    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make("FetchReach-v1")
        self.metadata = self.fetch_env.metadata
        self.hardcoded_controller = create_reach_controller_from_env(self.fetch_env)
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space#spaces.Box(low=-np.inf, high=np.inf, shape=(13,)) #self.fetch_env.observation_space

    def step(self, residual_action):
        # print("self.num_step:", self.num_step)
        self.num_step += 1
        if self._controller_done:
            controller_action = self._last_controller_action
        else:
            controller_action, controller_done = self.hardcoded_controller.get_action()

            if controller_done:
                self._last_controller_action = controller_action
                self._controller_done = True
        
        action = np.add(controller_action, residual_action)

        observation, reward, done, debug_info = self.fetch_env.step(action)
        self.hardcoded_controller.observe(observation, reward)
        print(action)
        # return np.hstack((observation['observation'], observation['desired_goal'])), reward, done, debug_info
        return observation, reward, done, debug_info

    def reset(self):
        obs = self.fetch_env.reset()
        self.num_step = 0
        self.hardcoded_controller.reset(obs)
        self._controller_done = False
        # return np.hstack((obs['observation'], obs['desired_goal']))
        return obs

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

class GeneralizedResidualFetchReachEnv(ResidualFetchReachEnv):

    def __init__(self, *args, **kwargs):
        super(GeneralizedResidualFetchReachEnv, self).__init__(*args, **kwargs)
        self.observation_space.spaces['observation'] = spaces.Box(low=np.hstack((self.fetch_env.action_space.low, self.fetch_env.observation_space.spaces['observation'].low)), 
            high=np.hstack((self.fetch_env.action_space.high, self.fetch_env.observation_space.spaces['observation'].high)),dtype=np.float32)

    def step(self, action):

        observation, reward, done, debug_info = self.fetch_env.step(action)
        self.hardcoded_controller.observe(observation, reward)
        if self._controller_done:
            controller_action = self._last_controller_action
        else:
            controller_action, controller_done = self.hardcoded_controller.get_action()

            if controller_done:
                self._last_controller_action = controller_action
                self._controller_done = True
        observation['observation'] = np.hstack((controller_action,observation['observation']))
        return observation, reward, done, debug_info

    def reset(self):
        obs = self.fetch_env.reset()
        self.num_step = 0
        self.hardcoded_controller.reset(obs)
        self._controller_done = False
        obs['observation'] = np.hstack((self.hardcoded_controller.get_action()[0],obs['observation']))
        return obs
