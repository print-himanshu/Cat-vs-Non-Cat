import initialization.two_layer_nn_initialization as two_layer
import initialization.l_layer_initialization as l_layer
import forward_propagation.linear_forward as fp_lf
import forward_propagation.linear_activation_forward as fp_activation
import forward_propagation.l_model_forward as forward_propagation
import cost.logistic_cost as cost
import backward_propagation.linear_backward as linear_b
import backward_propagation.l_model_backward as backward_propagation

import numpy as np

# 2 layer initialization parameters--------------------------------------------------
print("\n2 layer initialization test")
parameters = two_layer.initialize_parameters(3, 2, 1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# l layer initialization=-------------------------------------------------------------
print("\nL layer initialization")
parameters = l_layer.initialize_parameters_deep([5, 4, 3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# linear forward-------------------------------------------------------------------------
print("\nLinear forward")


def linear_forward_test_case():
    np.random.seed(1)

    A = np.array([[-1.02387576, 1.12397796, 0.74505627, 0.74505627],
                  [-1.62328545, 0.64667545, 0.74505627, 1.97611078],
                  [-1.74314104, -0.59664964, 0.74505627, -1.24412333],
                  ])

    W = np.array([[1.78862847,  0.43650985,  0.09649747],
                  [-1.864927, -0.2773882, -0.35475898],
                  [-0.0827414, -0.62700068, -0.04381817],
                  [-0.47721803, -1.31386475,  0.88462238]])

    b = np.array([[0.],
                  [0.],
                  [0.],
                  [0.]])

    # A = np.random.randn(3,2)
    # W = np.random.randn(1,3)
    # b = np.random.randn(1,1)

    return A, W, b


A, W, b = linear_forward_test_case()

Z, linear_cache = fp_lf.linear_forward(A, W, b)
print("Z = " + str(Z))


# Linear activation forward---------------------------------------------------------------------------
print("\nLinear Activation forward test case")


def linear_activation_forward_test_case():

    A_prev = np.array([[-1.02387576, 1.12397796, 0.74505627, 0.74505627],
                       [-1.62328545, 0.64667545, 0.74505627, 1.97611078],
                       [-1.74314104, -0.59664964, 0.74505627, -1.24412333],
                       ])

    W = np.array([[1.78862847,  0.43650985,  0.09649747],
                  [-1.864927, -0.2773882, -0.35475898],
                  [-0.0827414, -0.62700068, -0.04381817],
                  [-0.47721803, -1.31386475,  0.88462238]])
    b = np.array([[0.],
                  [0.],
                  [0.],
                  [0.]])

    # np.random.seed(2)
    # A_prev = np.random.randn(3,2)
    # W = np.random.randn(1,3)
    # b = np.random.randn(1,1)
    return A_prev, W, b


A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = fp_activation.linear_activation_forward(
    A_prev, W, b, activation="sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = fp_activation.linear_activation_forward(
    A_prev, W, b, activation="relu")
print("With ReLU: A = " + str(A))


# forward propagation------------------------------------------------------------------------
print("\nForward Propagation")


def L_model_forward_test_case():

    X = np.array([[-1.02387576, 1.12397796, 0.74505627, 0.74505627],
                  [-1.62328545, 0.64667545, 0.74505627, 1.97611078],
                  [-1.74314104, -0.59664964, 0.74505627, -1.24412333],
                  ])

    parameters = {'W1': np.array([[1.78862847,  0.43650985,  0.09649747],
                                  [-1.864927, -0.2773882, -0.35475898],
                                  [-0.0827414, -0.62700068, -0.04381817],
                                  [-0.47721803, -1.31386475,  0.88462238]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'W2': np.array([[0.88131804,  1.70957306,  0.05003364, -0.40467741],
                                  [-0.54535995, -1.54647732,
                                   0.98236743, -1.10106763],
                                  [-1.18504653, -0.2056499,  1.48614836,  0.23671627]]),
                  'W3': np.array([[-1.02378514, -0.7129932,  0.62524497],
                                  [-0.16051336, -0.76883635, -0.23003072]]),

                  'b2': np.array([[0.],
                                  [0.],
                                  [0.]]),

                  'b3': np.array([[0.],
                                  [0.]]),

                  'W4': np.array([[0.88131804,  1.70957306, ], ],),

                  'b4': np.array([[0.]]),

                  }

    # np.random.seed(1)
    # X = np.random.randn(4, 2)
    # W1 = np.random.randn(3, 4)
    # b1 = np.random.randn(3, 1)
    # W2 = np.random.randn(1, 3)
    # b2 = np.random.randn(1, 1)

    # parameters = {"W1": W1,
    #               "b1": b1,
    #               "W2": W2,
    #               "b2": b2}

    return X, parameters


X, parameters = L_model_forward_test_case()
AL, caches = forward_propagation.l_model_forward(X, parameters)
print("\nAL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


# Loss after 1 round------------------------------------------------------------
Y = np.asarray([[1, 1, 0, 1]])
print("\ncost = " + str(cost.compute_cost(AL, Y)))


# Linear Backward------------------------------------------------------------------
dZ, linear_cache = (np.array([[-0.8019545,  3.85763489]]),
                    (np.array([[-1.02387576,  1.12397796],
                               [-1.62328545,  0.64667545],
                               [-1.74314104, -0.59664964]]),
                     np.array([[0.74505627,  1.97611078, -1.24412333]]),
                     np.array([[1]],),),)

dA_prev, dW, db = linear_b.linear_backward(dZ, linear_cache)
print("\ndA_prev = " + str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db))


# Linear activation backward--------------------------------------------


# L model backward---------------------------------------------------------
print("\nBackward propagation")
AL = np.array([[0.5, 0.5, 0.5, 0.5]])
Y = np.array([[1, 1, 0, 1]])

grads = backward_propagation.l_model_backward(AL, Y, caches)
print(grads)
