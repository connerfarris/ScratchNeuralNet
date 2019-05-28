from sklearn.datasets import load_iris
from ScratchNN import *


print("Starting...")

num_classes = 3
iris_X = load_iris().data
iris_y = load_iris().target
np.random.seed(81)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]] / 16
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]] / 16
iris_y_test = iris_y[indices[-10:]]

# train
print("Training...")
training_targets = np.eye(num_classes)[iris_y_train]
nn = ScratchNN(inputs=iris_X_train, targets=training_targets, num_layers=4, num_nodes=[4, 100, 100, 3], activation_function=[None, "relu", "relu", "sigmoid"], cost_function="mean_squared")

nn.train(num_epochs=1000, learning_rate=0.1, filename="irissavepoint.pkl")
# nn.train_with_batches(batch_size=10, inputs=iris_X_train, labels=training_targets, num_epochs=10, learning_rate=0.1, filename="irissavepoint.pkl")

# for i in range(num_classes):
# 	print(nn.layers[i].error)

# test
print("Testing...")
# one_hot_targets_test = np.eye(num_classes)[iris_y_test]
# # nn.check_accuracy("irissavepoint.pkl", iris_X_test, one_hot_targets_test)
# print(iris_X_test)
# print(np.eye(num_classes)[iris_y_test])
nn.inputs = iris_X_test
nn.targets = np.eye(num_classes)[iris_y_test]
nn.forward_pass()
print(np.eye(num_classes)[iris_y_test])
print(nn.layers[3].activations)



