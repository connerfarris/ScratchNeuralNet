import numpy as np
import dill


class ScratchNN:

	def __init__(self, inputs, targets, num_layers, num_nodes, activation_function, cost_function):
		self.inputs = inputs
		self.targets = targets
		self.num_layers = num_layers
		self.num_nodes = num_nodes
		self.activation_function = activation_function
		self.layers = []
		self.cost_function = cost_function

		for i in range(num_layers - 1):
			layer_i = Layer(inputs.shape[0], num_nodes[i], num_nodes[i], num_nodes[i + 1], activation_function[i])
			self.layers.append(layer_i)

		last_layer = Layer(inputs.shape[0], targets.shape[1], 0, 0, activation_function[num_layers - 1])
		self.layers.append(last_layer)

	def train(self, num_epochs, learning_rate, filename):
		self.learning_rate = learning_rate
		for i in range(num_epochs):
			print("== EPOCH: ", i, " ==")
			self.error = 0
			self.forward_pass()
			self.calculate_error()
			self.back_pass()
			print("Error: ", self.error)
		dill.dump_session(filename)

	def train_with_batches(self, batch_size, inputs, labels, num_epochs, learning_rate, filename):
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		for j in range(num_epochs):
			i = 0
			print("== EPOCH: ", j, " ==")
			while i + batch_size != len(inputs):
				self.error = 0
				self.forward_pass(inputs[i:i + batch_size])
				self.calculate_error(labels[i:i + batch_size])
				self.back_pass(labels[i:i + batch_size])
				i = i + batch_size
			print("Error: ", self.error)
		dill.dump_session(filename)

	def forward_pass(self):
		self.layers[0].activations = self.inputs
		for i in range(self.num_layers - 1):
			temp1 = np.add(np.dot(self.layers[i].activations, self.layers[i].weights_for_layer), self.layers[i].biases_for_layer)
			temp2 = np.add(np.dot(self.layers[i].activations, self.layers[i].weights_for_layer), self.layers[i].biases_for_layer)
			if self.layers[i + 1].activation_function == "sigmoid":
				self.layers[i + 1].activations = self.sigmoid(temp1)
				self.layers[i + 1].activations_prime = self.sigmoidDerivative(temp2)
			elif self.layers[i + 1].activation_function == "relu":
				self.layers[i + 1].activations = self.relu(temp1)
				self.layers[i + 1].activations_prime = self.reluDerivative(temp2)
			elif self.layers[i + 1].activation_function == "leakyRelu":
				self.layers[i + 1].activations = self.leakyRelu(temp1)
				self.layers[i + 1].activations_prime = self.leakyReluDerivative(temp2)

	def calculate_error(self):
		if len(self.targets[0]) != self.layers[self.num_layers - 1].activation_columns:
			print("Error: Label is not of the same shape as output layer.")
			print("Label: ", len(self.targets), " : ", len(self.targets[0]))
			print("Out: ", len(self.layers[self.num_layers - 1].activations), " : ", len(self.layers[self.num_layers - 1].activations[0]))
			return

		if self.cost_function == "mean_squared":
			self.error = np.mean(
				np.divide(np.square(np.subtract(self.targets, self.layers[self.num_layers - 1].activations)), 2))
		elif self.cost_function == "cross_entropy":
			self.error = np.negative(np.sum(np.multiply(self.targets, np.log(self.layers[self.num_layers - 1].activations))))

	def back_pass(self):
		for i in range(self.num_layers - 1, 0, -1):
			if i == self.num_layers - 1:
				self.layers[i].error = np.multiply(2, self.layers[i].activations - self.targets)
				self.layers[i].delta = np.dot(self.layers[i - 1].activations.T, np.multiply(self.layers[i].error, self.layers[i].activations_prime))
				self.layers[i - 1].weights_for_layer -= np.multiply(self.learning_rate, self.layers[i].delta)
			else:
				self.layers[i].error = np.multiply(2, self.layers[i].activations - np.dot(self.layers[i + 1].activations, self.layers[i].weights_for_layer.T))
				self.layers[i].delta = np.dot(self.layers[i - 1].activations.T, np.multiply(self.layers[i].error, self.layers[i].activations_prime))
				self.layers[i - 1].weights_for_layer -= np.multiply(self.learning_rate, self.layers[i].delta)

	def predict(self, filename):
		dill.load_session(filename)
		# self.batch_size = 1
		self.forward_pass(self.inputs)
		a = self.layers[self.num_layers - 1].activations
		a[np.where(a == np.max(a))] = 1
		a[np.where(a != np.max(a))] = 0
		return a

	def check_accuracy(self, filename, inputs, targets):
		dill.load_session(filename)
		# self.batch_size = len(inputs)
		self.inputs = inputs
		self.targets = targets
		self.forward_pass()
		a = self.layers[self.num_layers - 1].activations
		a[np.where(a == np.max(a))] = 1
		a[np.where(a != np.max(a))] = 0
		total = 0
		correct = 0
		for i in range(len(a)):
			total = total + 1
			# print(self.layers[self.num_layers - 1].activations[i], a[i], labels[i])
			print(a[i], targets[i])
			if np.equal(a[i], targets[i]).all():
				correct = correct + 1
		# print("Accuracy: ", correct * 100 / total)
		print("Accuracy: " + str(correct) + "/" + str(total))

	def load_model(self, filename):
		dill.load_session(filename)

	def relu(self, layer):
		layer[layer < 0] = 0
		return layer

	def reluDerivative(self, layer):
		layer[layer <= 0] = 0
		layer[layer > 0] = 1
		return layer

	def leakyRelu(self, layer):
		out = np.where(layer > 0, layer, layer * 0.01)
		return out

	def leakyReluDerivative(self, layer):
		dx = np.ones_like(layer)
		dx[layer < 0] = 0.01
		return dx

	def sigmoid(self, layer):
		return np.divide(1, np.add(1, np.exp(layer)))

	def sigmoidDerivative(self, x):
		return x * (1.0 - x)


class Layer:
	def __init__(self, activation_rows, activation_columns, weight_rows, weight_columns, activation_function):
		self.activation_rows = activation_rows
		self.activation_columns = activation_columns
		self.weight_rows = weight_rows
		self.weight_columns = weight_columns
		self.activation_function = activation_function
		self.activations = np.zeros(shape=(activation_rows, activation_columns))
		self.activations_prime = np.zeros(shape=(activation_rows, activation_columns))
		self.weights_for_layer = np.random.normal(0, 0.001, size=(weight_rows, weight_columns))
		self.biases_for_layer = np.zeros(shape=(1, weight_columns))
		self.error = np.zeros(shape=(activation_rows, activation_columns))
		self.delta = np.atleast_2d()


def __str__(self):
	return 'NodeNum: ' + str(self.num_nodes_in_layer) + '\nActivationFunction: ' + str(self.activation_function) \
		   + '\nActivations Size: ' + str(self.activations.shape) + '\nWeights Size: ' \
		   + str(self.weights_for_layer.shape) \
		   + '\nActivations: ' + str(self.activations) + '\nWeights: ' + str(self.weights_for_layer) \
		   + str(self.biases_for_layer.shape) \
		   + '\nActivations: ' + str(self.activations) + '\nBiases: ' + str(self.biases_for_layer)
