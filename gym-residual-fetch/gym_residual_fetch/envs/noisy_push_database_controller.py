from scipy.spatial import KDTree

import os
import numpy as np

database_filename = 'noisy_push_database_500samples.npy'
dir_path = os.path.dirname(os.path.realpath(__file__))
database_file = os.path.join(dir_path, database_filename)

def create_noisy_push_database_controller(scale_by_distance=True, scale_factor=1.):

    states, actions = np.load(database_file)
    states = np.array([list(s) for s in states])
    kdtree = KDTree(states)

    def controller(obs):
        distance, idx = kdtree.query(obs['observation'])
        if scale_by_distance:
            return actions[idx] * 1./(1. + scale_factor * distance)
            
        return actions[idx]

    return controller
