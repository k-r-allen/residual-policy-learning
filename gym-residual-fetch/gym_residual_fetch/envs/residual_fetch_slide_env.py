from .slide_controller import get_slide_control

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import time



class ResidualFetchSlideEnv(gym.Env):

    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make("FetchSlide-v1")
        self.metadata = self.fetch_env.metadata
        self.hardcoded_controller = None
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space #spaces.Box(low=-np.inf, high=np.inf, shape=(28,)) #self.fetch_env.observation_space

    def step(self, residual_action):
        residual_action = 2. * residual_action
        action = np.add(residual_action, get_slide_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = self.fetch_env.step(action)
        self._last_observation = observation
        
        # return np.hstack((observation['observation'], observation['desired_goal'])), reward, done, debug_info
        return observation, reward, done, debug_info

    def reset(self):
        observation = self.fetch_env.reset()
        self._last_observation = observation
        # controller_action =  get_slide_control(observation)

        # return np.hstack((obs['observation'], obs['desired_goal']))
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

class GeneralizedResidualFetchSlideEnv(ResidualFetchSlideEnv):

    def __init__(self, *args, **kwargs):
        super(GeneralizedResidualFetchSlideEnv, self).__init__(*args, **kwargs)
        self.observation_space.spaces['observation'] = spaces.Box(low=np.hstack((self.fetch_env.action_space.low, self.fetch_env.observation_space.spaces['observation'].low)), 
            high=np.hstack((self.fetch_env.action_space.high, self.fetch_env.observation_space.spaces['observation'].high)),dtype=np.float32)

    def step(self, action):

        action = np.add(action, get_slide_control(self._last_observation))
        observation, reward, done, debug_info = self.fetch_env.step(action)
        self._last_observation = {k:v for k,v in observation.items()}
        controller_action = get_slide_control(observation)
        observation['observation'] = np.hstack((controller_action, observation['observation']))

        return observation, reward, done, debug_info

    def reset(self):
        observation = self.fetch_env.reset()
        self._last_observation = {k:v for k,v in observation.items()}
        controller_action =  get_slide_control(observation)
        observation['observation'] = np.hstack((controller_action, observation['observation']))

        return observation
