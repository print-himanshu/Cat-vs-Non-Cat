#    """
#     Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

#     Arguments:
#     AL -- probability vector, output of the forward propagation (L_model_forward())
#     Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
#     caches -- list of caches containing:
#                 every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
#                 the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

#     Returns:
#     grads -- A dictionary with the gradients
#              grads["dA" + str(l)] = ...
#              grads["dW" + str(l)] = ...
#              grads["db" + str(l)] = ...
#     """
import numpy as np
import backward_propagation.linear_activation_backward as linear_ac_b


def l_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # last layer calculation
    dAL = - np.divide(Y, AL) + np.divide((1-Y), (1-AL))
    current_cache = caches[L-1]

    # putting data of last layer

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)
                                                        ] = linear_ac_b.linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_ac_b.linear_activation_backward(grads["dA"+ str(l+1)], current_cache,"relu")
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
        grads["dA" + str(l)] = dA_prev_temp

    return grads