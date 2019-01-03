from gym.envs.robotics import rotations, fetch_env
from gym import utils, spaces
import numpy as np
import os

from .hook_controller import get_hook_control

import pdb

DIR_PATH = os.path.dirname(os.path.abspath(__file__))



class FetchHookEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, xml_file=None):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.35, 0.35, 0.4, 1., 0., 0., 0.],
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
            object_xpos_x = 1.8 + self.np_random.uniform(-0.05, 0.10)
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

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

class ResidualFetchHookEnv(FetchHookEnv):

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        self._last_observation = observation
        
        return observation, reward, done, debug_info

    def compute_reward(self, *args, **kwargs):
        return FetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation

        return observation

class NoisyFetchHookEnv(FetchHookEnv):

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]#object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

    def compute_reward(self, *args, **kwargs):
        return FetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation

        return observation

class TwoFrameResidualHookNoisyEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        super(TwoFrameResidualHookNoisyEnv, self).__init__()
        self.observation_space.spaces['observation'] = spaces.Box(low=np.hstack((self.observation_space.spaces['observation'].low, self.observation_space.spaces['observation'].low)), 
            high=np.hstack((self.observation_space.spaces['observation'].high, self.observation_space.spaces['observation'].high)),dtype=np.float32)
    
    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]#object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        obs_out = observation.copy()
        obs_out['observation'] = np.hstack((self._last_observation['observation'], observation['observation'])) 
        self._last_observation = observation
        
        return obs_out, reward, done, debug_info

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation.copy()
        observation['observation'] = np.hstack((self._last_observation['observation'], observation['observation']))
        return observation

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

class TwoFrameHookNoisyEnv(FetchHookEnv):
    def __init__(self, xml_file=None):
        super(TwoFrameHookNoisyEnv, self).__init__()
        self.observation_space.spaces['observation'] = spaces.Box(low=np.hstack((self.observation_space.spaces['observation'].low, self.observation_space.spaces['observation'].low)), 
            high=np.hstack((self.observation_space.spaces['observation'].high, self.observation_space.spaces['observation'].high)),dtype=np.float32)
    
    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]#object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def step(self, action):
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        obs_out = observation.copy()
        obs_out['observation'] = np.hstack((self._last_observation['observation'], observation['observation'])) 
        self._last_observation = observation
        
        return obs_out, reward, done, debug_info

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation.copy()
        observation['observation'] = np.hstack((self._last_observation['observation'], observation['observation']))
        return observation

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

class NoisyResidualFetchHookEnv(FetchHookEnv):

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = FetchHookEnv.step(self, action)
        
        self._last_observation = observation
        
        return observation, reward, done, debug_info

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]#object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

    def compute_reward(self, *args, **kwargs):
        return FetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = FetchHookEnv.reset(self)
        self._last_observation = observation

        return observation
