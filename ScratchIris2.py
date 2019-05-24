import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ScratchNN3 import *

# initialize
print("Initializing...")
iris_data = load_iris().data
iris_targets = load_iris().target
iris_one_hot_target = pd.get_dummies(iris_targets)
x_train, x_target, y_train, y_target = train_test_split(iris_data, iris_one_hot_target, test_size=0.1, random_state=20, shuffle=True)
y_train = np.asarray(y_train)
y_target = np.asarray(y_target)
num_classes = iris_one_hot_target.shape[1]
np.random.seed(81)
nodes = [4, 100, 100, 3]
sample_num = x_train.shape[0]
nn = ScratchNN3(sample_num, num_nodes_list=nodes, activation_function=[None, "relu", "relu", "sigmoid"], cost_function="mean_squared")
for x in range(len(nodes)):
	print('weights' + str(x) + ' shape: ' + str(nn.layers[x].weights.shape))
	print('weights_delta' + str(x) + ' shape: ' + str(nn.layers[x].weights_delta.shape))
	print('biases' + str(x) + ' shape: ' + str(nn.layers[x].biases.shape))
	print('biases_delta' + str(x) + ' shape: ' + str(nn.layers[x].biases_delta.shape))
	print('zed' + str(x) + ' shape: ' + str(nn.layers[x].zed.shape))
	print('zed_delta' + str(x) + ' shape: ' + str(nn.layers[x].zed_delta.shape))
	print('activations' + str(x) + ' shape: ' + str(nn.layers[x].activations.shape))
	print('activations_delta' + str(x) + ' shape: ' + str(nn.layers[x].activations_delta.shape))
	print('activation function: ' + str(nn.layers[x].activation_function))

# train
print("Training...")



# nn.train(num_epochs=1000, learning_rate=0.1, filename="irissavepoint.pkl")
# # nn.train_with_batches(batch_size=10, inputs=iris_X_train, labels=training_targets, num_epochs=10, learning_rate=0.1, filename="irissavepoint.pkl")
#
# # for i in range(num_classes):
# # 	print(nn.layers[i].error)
#
# # test
# print("Testing...")
# # one_hot_targets_test = np.eye(num_classes)[iris_y_test]
# # # nn.check_accuracy("irissavepoint.pkl", iris_X_test, one_hot_targets_test)
# # print(iris_X_test)
# # print(np.eye(num_classes)[iris_y_test])
# nn.inputs = iris_X_test
# nn.targets = np.eye(num_classes)[iris_y_test]
# nn.forward_pass()
# print(np.eye(num_classes)[iris_y_test])
# print(nn.layers[3].activations)



