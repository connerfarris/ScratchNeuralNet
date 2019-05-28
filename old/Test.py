# import os.path
# import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ScratchNN3 import *
from WeightsInitializer import *
import dill


# initialize
print("Initializing...")
iris_data = load_iris().data
iris_targets = load_iris().target
iris_one_hot_target = pd.get_dummies(iris_targets)
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_one_hot_target, test_size=0.1, random_state=20, shuffle=False)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
num_classes = iris_one_hot_target.shape[1]
np.random.seed(3)
nodes = [4, 128, 128, 3]
sample_num = x_train.shape[0]
nn = ScratchNN3(sample_num, num_nodes_list=nodes, activation_function=[None, "relu", "relu", "sigmoid"], cost_function="mean_squared")
dill.load_session("irissavepoint.pkl")
print(nn.layers[0].weights)

w = WeightsInitializer()
print(w.weights[0])



# nn = ScratchNN(4, [4, 16, 16, 3], [None, "relu", "relu", "relu"], cost_function="mean_squared")
# for layer in nn.layers:
# 	print('activations shape: ' + str(layer.activations.shape))
# 	print('activation function: ' + str(layer.activation_function))
# 	if layer.weights_for_layer is not None:
# 		print('weights shape: ' + str(layer.weights_for_layer.shape))
# 	else:
# 		print('weights shape: None')
# 	print(layer.activation_function)
# 	if layer.biases_for_layer is not None:
# 		print('biases shape: ' + str(layer.biases_for_layer.shape))
# 	else:
# 		print('biases shape: None')
# 	print('nodes in layer: ' + str(layer.num_nodes_in_layer))



# nn = ScratchNN(4, [4, 255, 128, 3], [None, "relu", "relu", "relu"], cost_function="mean_squared")
# layer0 = nn.layers.pop()
# layer1 = nn.layers.pop()
# print(layer0.activations.shape)
# # print(layer0.biases_for_layer.shape)
# # print(layer0.weights_for_layer.shape)
# print(layer1.activations.shape)
# print(layer1.biases_for_layer.shape)
# print(layer1.weights_for_layer.shape)
# # print(layer1.activations)

# a = [[1, 0], [0, 1]]
# a = np.asarray(a)
# b = [[4, 1], [2, 2]]
# b = np.asarray(b)
# c = np.dot(a, b)
# print(a)
# print(b)
# print(c)


#
# def sigmoid(Z):
#     A = 1 / (1 + np.exp(-Z))
#     return A, Z
#
#
# def tanh(Z):
#     A = np.tanh(Z)
#     return A, Z
#
#
# def relu(Z):
#     A = np.maximum(0, Z)
#     return A, Z
#
#
# def leaky_relu(Z):
#     A = np.maximum(0.1 * Z, Z)
#     return A, Z
#
#
# # Plot the 4 activation functions
# z = np.linspace(-10, 10, 100)
#
# # Computes post-activation outputs
# A_sigmoid, z = sigmoid(z)
# A_tanh, z = tanh(z)
# A_relu, z = relu(z)
# A_leaky_relu, z = leaky_relu(z)
#
# # Plot sigmoid
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.plot(z, A_sigmoid, label="Function")
# plt.plot(z, A_sigmoid * (1 - A_sigmoid), label = "Derivative")
# plt.legend(loc="upper left")
# plt.xlabel("z")
# plt.ylabel(r"$\frac{1}{1 + e^{-z}}$")
# plt.title("Sigmoid Function", fontsize=16)
# # Plot tanh
# plt.subplot(2, 2, 2)
# plt.plot(z, A_tanh, 'b', label = "Function")
# plt.plot(z, 1 - np.square(A_tanh), 'r',label="Derivative")
# plt.legend(loc="upper left")
# plt.xlabel("z")
# plt.ylabel(r"$\frac{e^z - e^{-z}}{e^z + e^{-z}}$")
# plt.title("Hyperbolic Tangent Function", fontsize=16)
# # plot relu
# plt.subplot(2, 2, 3)
# plt.plot(z, A_relu, 'g')
# plt.xlabel("z")
# plt.ylabel(r"$max\{0, z\}$")
# plt.title("ReLU Function", fontsize=16)
# # plot leaky relu
# plt.subplot(2, 2, 4)
# plt.plot(z, A_leaky_relu, 'y')
# plt.xlabel("z")
# plt.ylabel(r"$max\{0.1z, z\}$")
# plt.title("Leaky ReLU Function", fontsize=16)
# plt.tight_layout()
#
# plt.show()
#
#
#
#
