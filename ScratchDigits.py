import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from ScratchNN import *

# initialize
print("Initializing...")
digits_data = load_digits().data
digits_targets = load_digits().target
digits_one_hot_target = pd.get_dummies(digits_targets)
x_train, x_test, y_train, y_test = train_test_split(digits_data, digits_one_hot_target, test_size=0.1, random_state=20, shuffle=False)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
num_features = digits_data.shape[1]
num_classes = digits_one_hot_target.shape[1]
num_samples = x_train.shape[0]
nodes = [num_features, num_classes]
num_nodes_in_hidden_layers = [128, 128]
np.random.seed(75)
for x in range(len(num_nodes_in_hidden_layers)):
	nodes.insert(x + 1, num_nodes_in_hidden_layers[x])
print(nodes)

nn = ScratchNN(num_samples, num_nodes_list=nodes, activation_function=[None, "sigmoid", "sigmoid", "sigmoid"], cost_function="mean_squared")

# train
print("Training...")
nn.train(inputs=x_train, targets=y_train, num_epochs=1000, learning_rate=0.1, filename="digitssavepoint.pkl")

# test
print("Testing...")
nn.check_accuracy(filename="digitssavepoint.pkl", inputs=x_test, targets=y_test)



