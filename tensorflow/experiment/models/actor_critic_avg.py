import tensorflow as tf
from baselines.her.util import store_args
from .utils import nn, nn_last_zero
import copy
import pdb

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        o0 = o[:,:int(dimo/2)]
        o1 = o[:,int(dimo/2):]
        input0 = tf.concat(axis=1, values=[o0,g])
        input1 = tf.concat(axis=1, values=[o1,g])

        with tf.variable_scope('pi'):

            pi_tf0 = nn_last_zero(input0, [self.hidden] * self.layers + [self.dimu], seed_offset=0)
            pi_tf1 = nn_last_zero(input1, [self.hidden] * self.layers + [self.dimu], reuse=True, seed_offset=0)
            pi_tf = 0.5*(pi_tf0 + pi_tf1)
            self.pi_tf = self.max_u * tf.tanh(pi_tf)

        with tf.variable_scope('Q'):
            # for policy training
            input_Q0 = tf.concat(axis=1, values=[o0, g, self.pi_tf / self.max_u])
            input_Q1 = tf.concat(axis=1, values=[o1, g, self.pi_tf / self.max_u])
            Q_pi_tf0 = nn(input_Q0, [self.hidden] * self.layers + [1])
            Q_pi_tf1 = nn(input_Q1, [self.hidden]*self.layers + [1], reuse=True)
            self.Q_pi_tf = 0.5*(Q_pi_tf0 + Q_pi_tf1)

            # for critic training
            input_Q0 = tf.concat(axis=1, values=[o0, g, self.u_tf / self.max_u])
            input_Q1 = tf.concat(axis=1, values=[o1, g, self.u_tf / self.max_u])
            input_Q = 0.5*(input_Q0+input_Q1)
            self._input_Q = input_Q  # exposed for tests
            Q_tf1 = nn(input_Q1, [self.hidden]*self.layers + [1], reuse=True)
            Q_tf0=nn(input_Q0, [self.hidden] * self.layers + [1], reuse=True)
            self.Q_tf = 0.5*(Q_tf0 + Q_tf1)
            # self._Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True, use_seed=True)
            # Q_tf_copy = nn(input_Q, [self.hidden] * self.layers + [1], name='Q_tf_tomcopy', use_seed=True)
            # tf.stop_gradient(Q_tf_copy)

            # self.Q_tf = self._Q_tf - Q_tf_copy

