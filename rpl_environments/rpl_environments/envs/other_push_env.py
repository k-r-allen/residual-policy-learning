from .push_database_controller import create_push_database_controller

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.mujoco import mujoco_env

import mujoco_py
import numpy as np
import time
import pdb
import os

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal = np.zeros(3) # tmp
        self.distance_threshold = 0.05

        dir_path = os.path.dirname(os.path.realpath(__file__))
        fullpath = '%s/assets/pusher.xml' % dir_path
        frame_skip = 4

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation['observation'].size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()

        utils.EzPickle.__init__(self)

        self.reset_model()

        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(ob['achieved_goal'], self.goal),
            'tips_arm': self.get_body_com("tips_arm"), # hack for compute reward
            'action': a,
        }
        reward = self.compute_reward(ob['achieved_goal'], self.goal, info)

        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        a = info['action']
        tips_arm = info['tips_arm']
        obj_pos = achieved_goal
        
        vec_1 = obj_pos - tips_arm
        vec_2 = obj_pos - goal

        reward_near = -np.sum(np.abs(vec_1), axis=-1)
        reward_dist = -np.sum(np.abs(vec_2), axis=-1)
        reward_ctrl = -np.square(a).sum(axis=-1)
        reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        return reward

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.goal = self.get_body_com("goal")

        return self._get_obs()

    def _get_obs(self):
        object_observation = self.get_body_com("object")

        obs = np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            object_observation,
        ])

        observation = {
            'observation': obs,
            'achieved_goal': object_observation.copy(),
            'desired_goal': self.goal.copy(),
        }

        self._last_observation = observation

        return observation


class ResidualPusherEnv(PusherEnv):

    def __init__(self):
        self.controller = create_push_database_controller()
        self._last_observation = None
        PusherEnv.__init__(self)

    def step(self, residual_action):
        residual_action = 2. * residual_action

        obs = self._get_obs()

        action = np.add(residual_action, self.controller(obs))
        action = np.clip(action, -1, 1)
        
        return PusherEnv.step(self, action)


