import matplotlib.pyplot as plt
import initialization.l_layer_initialization as initialization
import forward_propagation.l_model_forward as forward_propagation
import backward_propagation.l_model_backward as backward_propagation
import cost.logistic_cost as lg_cost
import backward_propagation.update_parameters as update_parameters
import numpy as np


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    #     """
    # Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    # Arguments:
    # X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    # Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    # layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    # learning_rate -- learning rate of the gradient descent update rule
    # num_iterations -- number of iterations of the optimization loop
    # print_cost -- if True, it prints the cost every 100 steps
    
    # Returns:
    # parameters -- parameters learnt by the model. They can then be used to predict.
    # """

    
    #Creating variable
    costs = []
    print("L layer model")


    #Initialize parameters
    parameters = initialization.initialize_parameters_deep(layers_dims)


    #Looping (gradient descent)
    for i in range(0,num_iterations):
        #forward Propagation
        AL, caches = forward_propagation.l_model_forward(X, parameters)

        #Cost
        cost = lg_cost.compute_cost(AL, Y)

        #Backward Propagation
        grads = backward_propagation.l_model_backward(AL, Y, caches)


        #Update parameters
        parameters = update_parameters.update_parameters(parameters, grads, learning_rate)

        if print_cost and i%100 == 0:
            print("Cost after iteration {}: {}".format(i,cost))
            costs.append(cost)

    #Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel("Cost")
    plt.xlabel("Iteration (per Hundreds)")
    plt.title("Learning rate = "+str(learning_rate))
    plt.show()

    return parameters