import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from ScratchNN3 import *
from ScratchNN import *

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
#
# # train
# print("Training...")
# nn.train(inputs=x_train, targets=y_train, num_epochs=1000, learning_rate=0.1, filename="irissavepoint.pkl")
#
# # test
# print("Testing...")
# nn.check_accuracy(filename="irissavepoint.pkl", inputs=x_test, targets=y_test)


# initialize
print("Initializing...")
digits_data = load_digits().data
digits_targets = load_digits().target
digits_one_hot_target = pd.get_dummies(digits_targets)
d_x_train, d_x_test, d_y_train, d_y_test = train_test_split(digits_data, digits_one_hot_target, test_size=0.1, random_state=20, shuffle=False)
d_y_train = np.asarray(d_y_train)
d_y_test = np.asarray(d_y_test)
d_num_classes = digits_one_hot_target.shape[1]
np.random.seed(81)
d_nodes = [64, 500, 500, 10]
d_sample_num = d_x_train.shape[0]
d_nn = ScratchNNFinal(d_sample_num, num_nodes_list=d_nodes, activation_function=[None, "sigmoid", "sigmoid", "sigmoid"], cost_function="mean_squared")

# train
print("Training...")
d_nn.train(inputs=d_x_train, targets=d_y_train, num_epochs=1000, learning_rate=0.1, filename="digitssavepoint.pkl")

# test
print("Testing...")
d_nn.check_accuracy(filename="digitssavepoint.pkl", inputs=d_x_test, targets=d_y_test)



