from sklearn.datasets import load_iris
from ScratchNN import *


print("Starting...")

num_classes = 3
iris_X = load_iris().data
iris_y = load_iris().target
np.random.seed(84)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

print("Training...")

# train
one_hot_targets_train = np.eye(num_classes)[iris_y_train]
nn = ScratchNN(4, [4, 255, 128, 3], [None, "sigmoid", "tanh", "softmax"], cost_function="mean_squared")
nn.train(batch_size=1, inputs=iris_X_train, labels=one_hot_targets_train, num_epochs=100, learning_rate=0.001, filename="irissavepoint.pkl")

print("Testing...")

# test
one_hot_targets_test = np.eye(num_classes)[iris_y_test]
nn.check_accuracy("irissavepoint.pkl", iris_X_test, one_hot_targets_test)

