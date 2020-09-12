import numpy as np
from initialization import two_layer_nn_initialization as two_layer_int
import forward_propagation.linear_activation_forward as forward_activation
import backward_propagation.linear_activation_backward as backward_activation
import cost.logistic_cost as lg_cost
import matplotlib.pyplot as plt
import backward_propagation.update_parameters as update



def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    #     """
    # Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    # Arguments:
    # X -- input data, of shape (n_x, number of examples)
    # Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    # layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    # num_iterations -- number of iterations of the optimization loop
    # learning_rate -- learning rate of the gradient descent update rule
    # print_cost -- If set to True, this will print the cost every 100 iterations

    # Returns:
    # parameters -- a dictionary containing W1, W2, b1, and b2
    # """

    # Creating Variable
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims

    # Random Initialization
    parameters = two_layer_int.initialize_parameters(n_x, n_h, n_y)

    # Fetching variable from parameters dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Looping (gradient descent)
    for i in range(0, num_iterations):

        # Forward Propagation
        A1, cache1 = forward_activation.linear_activation_forward(
            X, W1, b1, "relu")
        A2, cache2 = forward_activation.linear_activation_forward(
            A1, W2, b2, "sigmoid")

        #Compute Cost
        cost = lg_cost.compute_cost(A2, Y)


        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))


        # Backward Propagation
        dA1, dW2, db2 = backward_activation.linear_activation_backward(
            dA2, cache2, "sigmoid")
        dA0, dW1, db1 = backward_activation.linear_activation_backward(
            dA1, cache1, "relu")

        # Setting grads
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Updating parameters
        parameters = update.update_parameters(parameters, grads, learning_rate)

        # Retrieving parameters
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    
    #Plot the Cost
    plt.plot(np.squeeze(costs))
    plt.ylabel("Cost")
    plt.xlabel("Iterations (per hundred)")
    plt.title("Learning Rate  = " + str(learning_rate))
    plt.show()

    return parameters
