from scipy.spatial import KDTree

import os
import numpy as np

database_filename = 'PusherEnv_database_500samples.npy'
dir_path = os.path.dirname(os.path.realpath(__file__))
database_file = os.path.join(dir_path, database_filename)

class DatabaseController(object):
    def __init__(self, kdtree, actions):
        self.kdtree = kdtree
        self.actions = actions

    def __call__(self, obs):
        distance, idx = self.kdtree.query(obs['observation'])
        return self.actions[idx]

    def __getstate__(self):
        return False

def create_push_database_controller(scale_by_distance=False, scale_factor=1.):

    states, actions = np.load(database_file)
    states = np.array([list(s) for s in states])
    kdtree = KDTree(states)

    return DatabaseController(kdtree, actions)
