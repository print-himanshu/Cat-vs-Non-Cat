# """
# Implement the RELU function.

# Arguments:
# Z -- Output of the linear layer, of any shape

# Returns:
# A -- Post-activation parameter, of the same shape as Z
# cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
# """

import numpy as np


def relu_forward(z):
    A = np.maximum(0, z)

    # checking
    assert(A.shape == z.shape)

    cache = z
    return A, cache


def relu_backward(dA, activation_cache):
    dZ = np.array(dA, copy=True)
    z = activation_cache
    dZ[z <= 0] = 0
    assert (dZ.shape == z.shape)

    return dZ
