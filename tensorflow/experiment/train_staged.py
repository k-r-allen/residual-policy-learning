import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import configs.config_standard_pickandplace as config
from rollout_controller import RolloutWorker
#from baselines.her.rollout import RolloutWorker

from baselines.her.util import mpi_fork
import pickle as pkl
import tensorflow as tf
import pdb
from subprocess import CalledProcessError
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.common.mpi_adam import MpiAdam
def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def broadcast_coinflip(comm, rank, policy, policy_pi_lr ):
   if rank == 0: # this is root
       policy.pi_lr = policy_pi_lr
       policy.pi_lr = comm.bcast(policy_pi_lr, root=0)
   else:
       policy.pi_lr = None
       policy.pi_lr = comm.bcast(policy_pi_lr, root=0)

def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, skip_training, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    thresh = kwargs['thresh']
    prev_losses = [0.0]
    losses = [0.0]
    actor_losses = [0.0]
    original_pi_lr = policy.pi_lr


    random_eps = rollout_worker.random_eps
    noise_eps = rollout_worker.noise_eps
    coin_flipping = False

    for epoch in range(n_epochs):
        if not kwargs['scratch'] and (epoch == 0 or abs(np.mean(losses) - np.mean(prev_losses)) > thresh or skip_training):
            policy_pi_lr = 0.
            coin_flipping = True
        else:
            policy_pi_lr = original_pi_lr
            coin_flipping = False

        broadcast_coinflip(MPI.COMM_WORLD, rank, policy, policy_pi_lr)

        prev_losses = losses

        # train
        if not skip_training:
            losses = []
            actor_losses = []
            rollout_worker.clear_history()
            for _ in range(n_cycles):
                rollout_worker.random_eps = random_eps
                rollout_worker.noise_eps = noise_eps
                if coin_flipping:
                    deterministic_rollouts = np.random.random() < 0.5
                    if deterministic_rollouts:
                        rollout_worker.random_eps = 0.0
                        rollout_worker.noise_eps = 0.0
                episode = rollout_worker.generate_rollouts()
 
                policy.store_episode(episode)
                for _ in range(n_batches):
                    critic_loss, actor_loss = policy.train()
                    losses.append(actor_loss)
                    actor_losses.append(critic_loss)
                policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, skip_training, config_path,
    override_params={}, save_policies=True, policy_path=None
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    with open(config_path) as f:
        params = json.load(f)
    if 'TwoFrame' in env:
        params['network_class'] = 'models.actor_critic_avg:ActorCritic'

    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    #
    if policy_path is None:
        policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    if policy_path is not None:
        her_transits = config.get_her_transitions(dims=dims, params=params, clip_return=clip_return)
        policy = load_policy(policy_path, her_transits, freeze=False)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma','noise_eps', 'random_eps','controller_prop']: #['T', 'rollout_batch_size', 'gamma', 'controller_prop','noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    kwargs = {}
    if 'Residual' not in env:
        kwargs['scratch'] = True
    else:
        kwargs['scratch'] = False

    kwargs['thresh'] = params['threshold']

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies, skip_training=skip_training, **kwargs)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--skip_training', is_flag=True, help='whether or not training should be skipped')
@click.option('--policy_path', type=str, default=None, help='path to saved policy')
@click.option('--config_path', type=str, default=None, help='path to json config file')
@click.option('--random_eps', type=float, default=None, help='random_eps if using expert explore baseline')
@click.option('--controller_prop', type=float, default=None, help='alpha if using expert explore baseline')
def main(**kwargs):
    override_params = {}
    if kwargs['random_eps'] is not None:
        override_params['random_eps'] = kwargs['random_eps']
    if kwargs['controller_prop'] is not None:
        override_params['controller_prop'] = kwargs['controller_prop']
    
    kwargs['override_params'] = override_params
    kwargs.pop('random_eps',None)
    kwargs.pop('controller_prop',None)
    launch(**kwargs)


if __name__ == '__main__':
    main()
