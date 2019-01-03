from .mpc_controller import MPCController

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import copy
import numpy as np
# import pybullet as p
import time

class MPCPushEnv(gym.Env):

    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make("FetchPush-v1")
        self.metadata = self.fetch_env.metadata
        self.action_space = self.fetch_env.action_space
        self.action_scale = 10.
        self.observation_space = self.fetch_env.observation_space

    def step(self, action):
        observation, reward, done, debug_info = self.fetch_env.step(self.action_scale * action)
        sim_state = copy.deepcopy(self.fetch_env.env.sim.get_state())
        observation['observation'] = np.concatenate((observation['observation'], sim_state.qpos, sim_state.qvel))
        return observation, reward, done, debug_info

    def reset(self):
        observation = self.fetch_env.reset()
        sim_state = copy.deepcopy(self.fetch_env.env.sim.get_state())
        observation['observation'] = np.concatenate((observation['observation'], sim_state.qpos, sim_state.qvel))
        return observation

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
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

class ResidualMPCPushEnv(gym.Env):

    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make("FetchPush-v1")
        self.metadata = self.fetch_env.metadata
        self.action_space = self.fetch_env.action_space
        self.action_scale = 10.
        self.observation_space = self.fetch_env.observation_space

        self.hardcoded_controller = MPCController(gym.make("FetchPush-v1"), self.action_scale)

    def step(self, residual_action):
        residual_action = 2. * residual_action
        mj_sim_state = self.fetch_env.env.sim.get_state()
        qpos = mj_sim_state.qpos.copy()
        qvel = mj_sim_state.qvel.copy()
        sim_state = {'qpos' : qpos, 'qvel' : qvel}
        controller_action = self.hardcoded_controller.act(self._last_observation, {'sim_state' : sim_state})
        action = np.add(residual_action, controller_action)
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = self.fetch_env.step(self.action_scale * action)
        self._last_observation = observation

        return observation, reward, done, debug_info

    def reset(self):
        observation = self.fetch_env.reset()
        self._last_observation = observation

        return observation

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        self.hardcoded_controller.seed(seed=seed)
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




