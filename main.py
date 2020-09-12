import data_import as data
import matplotlib.pyplot as plt
import model.two_layer as two_layer
import model.l_layer as l_layer
import prediction.post_function as pdt

# Loading Data and knowing Dimension
train_x_origin, train_y, test_x_origin, test_y, classes = data.load_data()
m_train, num_px, m_test = data.gettingDimension(train_x_origin, test_x_origin)
train_x, test_x = data.flattening_data(train_x_origin, test_x_origin)


# Visualizing one test examples
index = 10
plt.imshow(train_x_origin[index])
plt.show()
print("\ny = " + str(train_y[0, index]) + ". It's a " +
      classes[train_y[0, index]].decode("utf-8") + " picture.")

# 2 Layer Neural Network (basic)
n_x = train_x.shape[0]
n_h = 7
n_y = 1
parameters = two_layer.two_layer_model(train_x, train_y, layer_dims=(
    n_x, n_h, n_y), num_iterations=2500, print_cost=True)

predictions_train = pdt.predict(train_x, train_y, parameters)
predictions_test = pdt.predict(test_x, test_y, parameters)


# L layer Neural Network
print("\n train_x.shape[0] : {}".format(train_x.shape[0]))
layers_dims = [train_x.shape[0], 20, 7, 5, 1]
parameters = l_layer.L_layer_model(
    train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
predictions_train = pdt.predict(train_x, train_y, parameters)
predictions_test = pdt.predict(test_x, test_y, parameters)
