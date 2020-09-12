    # """
    # Arguments:
    # layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    # Returns:
    # parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    #                 Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    #                 bl -- bias vector of shape (layer_dims[l], 1)
    # """

import numpy as np

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))

        #Checking dimension
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    
    return parameters

        