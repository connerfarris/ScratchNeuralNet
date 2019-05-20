from sklearn.datasets import load_iris
from ScratchNN import *


print("Starting...")

num_classes = 3
iris_X = load_iris().data
iris_y = load_iris().target
np.random.seed(3)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# print("Training...")

# train
training_targets = np.eye(num_classes)[iris_y_train]
nn = ScratchNN(4, [4, 10, 10, 3], [None, "relu", "relu", "sigmoid"], cost_function="mean_squared")

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

nn.train(batch_size=1, inputs=iris_X_train, labels=training_targets, num_epochs=1, learning_rate=1, filename="irissavepoint.pkl")

print("Testing...")

# test
one_hot_targets_test = np.eye(num_classes)[iris_y_test]
nn.check_accuracy("irissavepoint.pkl", iris_X_test, one_hot_targets_test)

