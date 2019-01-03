import tensorflow as tf
import numpy as np

def nn(input, layers_sizes, reuse=None, flatten=False, name="", use_seed=False, init_zero=False, seed_offset=0):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        if init_zero:
            kernel_init = 'zeros'
        elif use_seed:
            kernel_init = tf.contrib.layers.xavier_initializer(seed=i+seed_offset)
        else:
            kernel_init = tf.contrib.layers.xavier_initializer()
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=kernel_init,
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input

def nn_last_zero(input, layers_sizes, reuse=None, flatten=False, name="", seed_offset=0):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        if i == len(layers_sizes) - 1:
            initializer = 'zeros'
        else:
            initializer = tf.contrib.layers.xavier_initializer(seed=i+seed_offset)

        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=initializer,
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input

def nn_controller(controller_x, input, layers_sizes, dimu, max_u, reuse=None, flatten=False, name=""):
 
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])

    input = tf.layers.dense(inputs=tf.concat(axis=1, values=[controller_x, max_u * tf.tanh(input)]),
                            units=dimu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            reuse=reuse,
                            name=name+'_linear')
    return input

def nn_linear(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input