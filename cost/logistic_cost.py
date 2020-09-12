# """
# Implement the cost function defined by equation (7).

# Arguments:
# AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
# Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

# Returns:
# cost -- cross-entropy cost
# """
import numpy as np


def compute_cost(AL, Y):
    m = Y.shape[1]

    # cost = (-1./m) * np.sum(np.multiply(Y, np.log(AL)) +
    #                        np.multiply((1-Y), np.log(1-AL)))

    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost
