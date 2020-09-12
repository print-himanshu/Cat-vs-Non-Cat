# """
# Implement the forward propagation for the LINEAR->ACTIVATION layer

# Arguments:
# A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
# W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
# b -- bias vector, numpy array of shape (size of the current layer, 1)
# activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

# Returns:
# A -- the output of the activation function, also called the post-activation value
# cache -- a python tuple containing "linear_cache" and "activation_cache";
#          stored for computing the backward pass efficiently
# """

import activation_function.relu as relu
import activation_function.sigmoid as sigmoid
import forward_propagation.linear_forward as fp


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        z, linear_cache = fp.linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid.sigmoid_forward(z)

    elif activation == "relu":
        z, linear_cache = fp.linear_forward(A_prev, W, b)
        A, activation_cache = relu.relu_forward(z)

    # checking dimension
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache
