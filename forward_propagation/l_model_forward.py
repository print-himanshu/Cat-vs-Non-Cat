    # """
    # Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    # Arguments:
    # X -- data, numpy array of shape (input size, number of examples)
    # parameters -- Weight and bias unit of each layer are stored
    
    # Returns:
    # AL -- last post-activation value
    # caches -- list of caches containing:
    #             every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    # """

import forward_propagation.linear_activation_forward as fp
import numpy as np

def l_model_forward(X,parameters):

    caches = []
    A = X
    
    #Parameters have both W and b of each layer
    #layer count will be half of the total parameter count
    L = len(parameters)//2

    for l in range(1,L):
        A_prev = A
        A , cache = fp.linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b" + str(l)], "relu" )
        caches.append(cache)

    AL, cache = fp.linear_activation_forward(A, parameters["W" + str(L)] , parameters["b" + str(L)] , "sigmoid")
    caches.append(cache)


    #Checking dimension
    assert(AL.shape == (1,X.shape[1]))

    return AL, caches
