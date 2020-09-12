# """
# Implements the sigmoid activation in numpy

# Arguments:
# Z -- numpy array of any shape

# Returns:
# A -- output of sigmoid(z), same shape as Z
# cache -- returns Z as well, useful during backpropagation
# """

import numpy as np


def sigmoid_forward(x):
    a = 1/(1 + np.exp(-x))
    cache = x
    return a, cache


def sigmoid_backward(dA, activation_cache):
    z = activation_cache
    s = 1/(1 + np.exp(-z))
    dz = dA * s * (1-s)
    assert (dz.shape == z.shape)

    return dz
