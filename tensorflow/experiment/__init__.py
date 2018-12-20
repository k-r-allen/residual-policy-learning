import sys

# :'(
from . import config_copy
from . import models
from . import ddpg_controller
# from . import ddpg_controller_residual_base
sys.modules['config_copy'] = config_copy
sys.modules['models'] = models
sys.modules['ddpg_controller'] = ddpg_controller
# sys.modules['ddpg_controller_residual_base'] = ddpg_controller_residual_base

import pickle

def load_her_policy(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
