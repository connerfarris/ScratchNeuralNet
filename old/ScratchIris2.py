import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ScratchNN3 import *
from WeightsInitializer import *

w = WeightsInitializer()

# # initialize
# print("Initializing...")
# iris_data = load_iris().data
# iris_targets = load_iris().target
# iris_one_hot_target = pd.get_dummies(iris_targets)
# x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_one_hot_target, test_size=0.1, random_state=20, shuffle=False)
# y_train = np.asarray(y_train)
# y_test = np.asarray(y_test)
# num_classes = iris_one_hot_target.shape[1]
# np.random.seed(3)
# nodes = [4, 128, 128, 3]
# sample_num = x_train.shape[0]
# nn = ScratchNN3(sample_num, num_nodes_list=nodes, activation_function=[None, "sigmoid", "sigmoid", "sigmoid"], cost_function="mean_squared")

nn = ScratchNN3(w.sample_num, num_nodes_list=w.nodes, activation_function=[None, "sigmoid", "sigmoid", "sigmoid"], cost_function="mean_squared")

for x in range(len(w.nodes)):
	print('weights' + str(x) + ' shape: ' + str(nn.layers[x].weights.shape))
	# print('weights_delta' + str(x) + ' shape: ' + str(nn.layers[x].weights_delta.shape))
	print('biases' + str(x) + ' shape: ' + str(nn.layers[x].biases.shape))
	# print('biases_delta' + str(x) + ' shape: ' + str(nn.layers[x].biases_delta.shape))
	print('zed' + str(x) + ' shape: ' + str(nn.layers[x].zed.shape))
	print('zed_delta' + str(x) + ' shape: ' + str(nn.layers[x].zed_delta.shape))
	print('activations' + str(x) + ' shape: ' + str(nn.layers[x].activations.shape))
	print('activations_delta' + str(x) + ' shape: ' + str(nn.layers[x].activations_delta.shape))
	print('activation function: ' + str(nn.layers[x].activation_function))

# train
print("Training...")
nn.train(inputs=w.x_train, targets=w.y_train, num_epochs=1000, learning_rate=0.1, filename="irissavepoint.pkl")

# test
print("Testing...")
nn.check_accuracy(filename="irissavepoint.pkl", inputs=w.x_test, targets=w.y_test)



