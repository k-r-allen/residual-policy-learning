import numpy as np

from gym import utils, spaces
from gym.utils import seeding
from gym.envs.robotics import rotations, fetch_env
import gym

import mujoco_py

import glob
import glfw
import os
import pdb

from .hook_controller import get_hook_control



DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class ComplexHookSingleObjectEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, xml_file=None):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.6, 1., 0., 0., 0.],
            'hook:joint': [1.35, 0.35, 0.6, 1., 0., 0., 0.],
        }

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets', 'hook.xml')

        self._goal_pos = np.array([1.65, 0.75, 0.42])
        self._object_xpos = np.array([1.8, 0.75])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        
        utils.EzPickle.__init__(self)

    def render(self, mode="human", *args, **kwargs):
        # See https://github.com/openai/gym/issues/1081
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer().render()
            width, height = 3350, 1800
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

        return self.render(*args, **kwargs)

    def _sample_goal(self):
        goal_pos = self._goal_pos.copy()
        goal_pos[:2] += self.np_random.uniform(-0.05, 0.05)
        return goal_pos

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -24.

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        object_xpos_x = 1.65 + self.np_random.uniform(-0.05, 0.05)
        while True:
            object_xpos_x = 1.8 + self.np_random.uniform(-0.05, 0.05)
            object_xpos_y = 0.75 + self.np_random.uniform(-0.05, 0.05)
            if (object_xpos_x - self._goal_pos[0])**2 + (object_xpos_y - self._goal_pos[1])**2 >= 0.01:
                break
        self._object_xpos = np.array([object_xpos_x, object_xpos_y])

        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = self._object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])

        return obs


class ResidualComplexHookSingleObjectEnv(ComplexHookSingleObjectEnv):
    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = ComplexHookSingleObjectEnv.step(self, action)
        
        self._last_observation = observation
        
        return observation, reward, done, debug_info

    def compute_reward(self, *args, **kwargs):
        return ComplexHookSingleObjectEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = ComplexHookSingleObjectEnv.reset(self)
        self._last_observation = observation

        return observation

class TwoFrameResidualComplexHookSingleObjectEnv(ComplexHookSingleObjectEnv):
    def __init__(self, xml_file):
        super(TwoFrameResidualComplexHookSingleObjectEnv, self).__init__(xml_file)
        self.observation_space.spaces['observation'] = spaces.Box(low=np.hstack((self.observation_space.spaces['observation'].low, self.observation_space.spaces['observation'].low)), 
            high=np.hstack((self.observation_space.spaces['observation'].high, self.observation_space.spaces['observation'].high)),dtype=np.float32)

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = ComplexHookSingleObjectEnv.step(self, action)
        
        obs_out = observation.copy()
        obs_out['observation'] = np.hstack((self._last_observation['observation'], observation['observation'])) 
        self._last_observation = observation
        
        return obs_out, reward, done, debug_info

    def reset(self):
        observation = ComplexHookSingleObjectEnv.reset(self)
        self._last_observation = observation.copy()
        observation['observation'] = np.hstack((self._last_observation['observation'], observation['observation']))
        return observation

class ComplexHookEnv(gym.Env):
    def __init__(self, train=True, num_scenes=100, scene_offset=0, subenv_type='normal'):
        if train:
            xml_files = list(glob.glob(os.path.join(DIR_PATH, 'assets', 'fetch_complex_objects', 'fetch',
                'train_scene_hook_*.xml')))
        else:
            xml_files = list(glob.glob(os.path.join(DIR_PATH, 'assets', 'fetch_complex_objects', 'fetch',
                'test_scene_hook_*.xml')))

        xml_files = xml_files[scene_offset:scene_offset+num_scenes]

        self._subenvs = []

        self.seed()

        for xml_file in xml_files:
            try:
                if subenv_type == 'residual':
                    subenv = ResidualComplexHookSingleObjectEnv(xml_file)
                elif subenv_type == 'twoframe_residual':
                    subenv = TwoFrameResidualComplexHookSingleObjectEnv(xml_file)
                elif subenv_type == 'normal':
                    subenv = ComplexHookSingleObjectEnv(xml_file)
                else:
                    print('subenv_type not recognized')
                self._subenvs.append(subenv)
            except:
                print("FAILED; skipping")

        self._num_envs = len(self._subenvs)

        self.action_space = self._subenvs[0].action_space
        self.observation = self._subenvs[0].observation_space
        self.metadata = self._subenvs[0].metadata

    def reset(self):
        env_id = self.np_random.randint(self._num_envs)
        self._env = self._subenvs[env_id]
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)

        for subenv in self._subenvs:
            subenv.seed(seed=self.np_random.randint(1e8))

        return [seed]

    def compute_reward(self, *args, **kwargs):
        return self._env.compute_reward(*args, **kwargs)


class ComplexHookTrainEnv(ComplexHookEnv):
    def __init__(self):
        ComplexHookEnv.__init__(self, train=True)

class ResidualComplexHookTrainEnv(ComplexHookEnv):
    def __init__(self):
        ComplexHookEnv.__init__(self, train=True, subenv_type='residual')

class ComplexHookTestEnv(ComplexHookEnv):
    def __init__(self):
        ComplexHookEnv.__init__(self, train=False)

class ResidualComplexHookTestEnv(ComplexHookEnv):
    def __init__(self):
        ComplexHookEnv.__init__(self, train=False, subenv_type='residual')


class TwoFrameResidualComplexHookTrainEnv(ComplexHookEnv):
    def __init__(self):
        ComplexHookEnv.__init__(self, train=True, subenv_type='twoframe_residual')

class TwoFrameResidualComplexHookTestEnv(ComplexHookEnv):
    def __init__(self):
        ComplexHookEnv.__init__(self, train=False, subenv_type='twoframe_residual')
