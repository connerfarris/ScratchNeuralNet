import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ScratchNN import *

# manual inputs
num_nodes_in_hidden_layers = [500, 500, 500]
activation_function = [None, "sigmoid", "sigmoid", "sigmoid", "sigmoid"]
np.random.seed(75)

# initialize
print("Initializing...")
iris_data = load_iris().data
iris_targets = load_iris().target
iris_one_hot_target = pd.get_dummies(iris_targets)
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_one_hot_target, test_size=0.1, random_state=20, shuffle=False)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
num_features = iris_data.shape[1]
num_classes = iris_one_hot_target.shape[1]
num_samples = x_train.shape[0]
nodes = [num_features, num_classes]
for x in range(len(num_nodes_in_hidden_layers)):
	nodes.insert(x + 1, num_nodes_in_hidden_layers[x])
print(nodes)

nn = ScratchNN(num_samples, num_nodes_list=nodes, activation_function=activation_function, cost_function="mean_squared")

# train
print("Training...")
nn.train(inputs=x_train, targets=y_train, num_epochs=5000, learning_rate=0.1, filename="irissavepoint.pkl")

# test
print("Testing...")
nn.check_accuracy(filename="irissavepoint.pkl", inputs=x_test, targets=y_test)



