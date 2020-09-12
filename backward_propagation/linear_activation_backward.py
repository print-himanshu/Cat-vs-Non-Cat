    # """
    # Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    # Arguments:
    # dA -- post-activation gradient for current layer l 
    # cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    # activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    # Returns:
    # dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    # dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    # db -- Gradient of the cost with respect to b (current layer l), same shape as b
    # """

import backward_propagation.linear_backward as linear_b
import activation_function.sigmoid as sigmoid
import activation_function.relu as relu

def linear_activation_backward(dA,cache,activation):
    linear_cache, activation_cache = cache
    
    if activation == "sigmoid":
        dZ = sigmoid.sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_b.linear_backward(dZ, linear_cache)

    elif activation == "relu":
        dZ = relu.relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_b.linear_backward(dZ,linear_cache)

    return dA_prev, dW, db