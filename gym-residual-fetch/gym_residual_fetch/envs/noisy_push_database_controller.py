from .push_database_controller import DatabaseController
from scipy.spatial import KDTree

import os
import numpy as np

database_filename = 'NoisyPusherEnv_database_500samples.npy'
dir_path = os.path.dirname(os.path.realpath(__file__))
database_file = os.path.join(dir_path, database_filename)

def create_noisy_push_database_controller(scale_by_distance=False, scale_factor=1.):

    states, actions = np.load(database_file)
    states = np.array([list(s) for s in states])
    kdtree = KDTree(states)

    return DatabaseController(kdtree, actions)
