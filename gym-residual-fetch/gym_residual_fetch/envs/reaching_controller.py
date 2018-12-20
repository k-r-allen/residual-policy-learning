import math
import numpy as np

class ReachController():

    def get_action(self):
        current_pos = self.last_observation['observation'][:3]
        goal = self.last_observation['desired_goal']
        action = 10. * np.subtract(goal, current_pos)
        action = np.hstack((action, 0.))
        return action, False

    def observe(self, observation, reward):
        self.last_observation = observation

    def reset(self, observation):
        self.num_steps = 0
        self.last_observation = observation


def create_reach_controller_from_env(env):
    return ReachController()

