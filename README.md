# Residual Policy Learning
## Tom Silver\*, Kelsey Allen\*, Josh Tenenbaum, Leslie Kaelbling

## Abstract
We present Residual Policy Learning (RPL): a simple method for improving nondifferentiable policies using model-free deep reinforcement learning. RPL thrives in complex robotic manipulation tasks where good but imperfect controllers are available. In these tasks, reinforcement learning from scratch remains data-inefficient or intractable, but learning a *residual* on top of the initial controller can yield substantial improvements. We study RPL in six challenging MuJoCo tasks involving partial observability, sensor noise, model misspecification, and controller miscalibration. For initial controllers, we consider both hand-designed policies and model-predictive controllers with known or learned transition models. By combining learning with control algorithms, RPL can perform long-horizon, sparse-reward tasks for which reinforcement learning alone fails. Moreover, we find that RPL consistently and substantially improves on the initial controllers. We argue that RPL is a promising approach for combining the complementary strengths of deep reinforcement learning and robotic control, pushing the boundaries of what either can achieve independently.

## arXiv
https://arxiv.org/abs/1812.06298

## Website
https://k-r-allen.github.io/residual-policy-learning/

## System Requirements
We use Python 3.5.6 on Ubuntu 18.04 and macOS High Sierra. Other setups may work but have not been tested.

## Using the Environments Only

### Installation
First follow [the instructions](https://github.com/openai/mujoco-py) to install MuJoCo (mjpro 150) and mujoco-py. 

We had issues on some machines installing mujoco-py with pip. These issues were resolved by installing from source. To match our version exactly, you can do
```
pip install -e git+https://github.com/openai/mujoco-py@a9f563cbb81d45f2379c6bcc4a4cd73fac09c4be#egg=mujoco_py
```

Next:
```
cd rpl_environments
pip install -e .
```

*Note*: the `ComplexHook` environment requires a large zip file with MuJoCo xml and stl files. To use this environment, please download [fetch_complex_objects.zip](http://web.mit.edu/tslvr/www/fetch_complex_objects.zip) and unzip it in `rpl_environments/rpl_environments/envs/assets` (replacing the existing directory with the same name). The file is about 718 MB. For example:

```
cd rpl_environments/rpl_environments/envs/assets
rm -rf fetch_complex_objects
wget http://web.mit.edu/tslvr/www/fetch_complex_objects.zip
unzip fetch_complex_objects.zip
```

### Usage Examples
```
import gym
import rpl_environments

for env_name in ["SlipperyPush-v0", "FetchHook-v0", "TwoFrameHookNoisy-v0", "ComplexHook-v0"]:
    env = gym.make(env_name)
    obs = env.reset()
    env.render()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, debug_info = env.step(action)
        env.render()
```

## Reproducing the Experiments

### Install
First follow the installation instructions for the RPL environments above.

Next visit [https://github.com/openai/baselines](https://github.com/openai/baselines) and install their prerequisites. To match our version of baselines exactly, you can do:

```
pip install -e git+https://github.com/openai/baselines.git@c28acb22030f594f94d128bf47b489cc704f593e#egg=baselines
```

To use the plotting code, you will also need matplotlib, pandas, and seaborn.

### Usage Examples

See `tensorflow/experiments/run_all_experiments.sh`.
