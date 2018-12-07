from sklearn.datasets import load_digits
from ScratchNN import *


print("Starting...")

num_classes = 10
digits_X = load_digits().data
digits_y = load_digits().target

np.random.seed(84)
indices = np.random.permutation(len(digits_X))
digits_X_train = digits_X[indices[:-100]]
digits_y_train = digits_y[indices[:-100]]
digits_X_test = digits_X[indices[-100:]]
digits_y_test = digits_y[indices[-100:]]

print("Training...")

# train
one_hot_targets_train = np.eye(num_classes)[digits_y_train]
nn = ScratchNN(num_layers=4, num_nodes=[digits_X.shape[1], 100, 50, num_classes], activation_function=[None, "sigmoid", "sigmoid", "softmax"], cost_function="mean_squared")
nn.train(batch_size=1, inputs=digits_X_train, labels=one_hot_targets_train, num_epochs=100, learning_rate=0.001, filename="irissavepoint.pkl")

# print("Testing...")
#
# # test
# one_hot_targets_test = np.eye(num_classes)[iris_y_test]
# nn.check_accuracy("irissavepoint.pkl", iris_X_test, one_hot_targets_test)

