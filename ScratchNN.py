import numpy as np
import dill


class ScratchNN:

	def __init__(self, inputs, targets, num_layers, num_nodes, activation_function, cost_function):
		self.inputs = inputs
		self.targets = targets
		self.num_layers = num_layers
		self.num_nodes = num_nodes
		self.layers = []
		self.cost_function = cost_function

		for i in range(num_layers - 1):
			layer_i = Layer(inputs.shape[0], num_nodes[i], num_nodes[i], num_nodes[i + 1], activation_function[i])
			self.layers.append(layer_i)

	def train(self, num_epochs, learning_rate, filename):
		self.learning_rate = learning_rate
		for i in range(num_epochs):
			print("== EPOCH: ", i, " ==")
			self.error = 0
			self.forward_pass(self.inputs)
			self.calculate_error(self.targets)
			self.back_pass(self.targets)

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

	def forward_pass(self, inputs):
		self.layers[0].activations = inputs
		for i in range(self.num_layers - 1):
			temp = np.add(np.dot(self.layers[i].activations, self.layers[i].weights_for_layer),
						  self.layers[i].biases_for_layer)
			if self.layers[i + 1].activation_function == "sigmoid":
				self.layers[i + 1].activations = self.sigmoid(temp)
			elif self.layers[i + 1].activation_function == "relu":
				self.layers[i + 1].activations = self.relu(temp)
			elif self.layers[i + 1].activation_function == "leakyRelu":
				self.layers[i + 1].activations = self.leakyRelu(temp)

	def calculate_error(self, labels):
		if len(labels[0]) != self.layers[self.num_layers - 1].num_nodes_in_layer:
			print("Error: Label is not of the same shape as output layer.")
			print("Label: ", len(labels), " : ", len(labels[0]))
			print("Out: ", len(self.layers[self.num_layers - 1].activations), " : ",
				  len(self.layers[self.num_layers - 1].activations[0]))
			return

		if self.cost_function == "mean_squared":
			self.error = np.mean(
				np.divide(np.square(np.subtract(labels, self.layers[self.num_layers - 1].activations)), 2))
		elif self.cost_function == "cross_entropy":
			self.error = np.negative(np.sum(np.multiply(labels, np.log(self.layers[self.num_layers - 1].activations))))

	def back_pass(self, labels):
		# if self.cost_function == "cross_entropy" and self.layers[self.num_layers-1].activation_function == "softmax":
		targets = labels
		i = self.num_layers - 1
		y = self.layers[i].activations
		# deltaw = np.matmul(np.asarray(self.layers[i - 1].activations).T, np.multiply(y, np.multiply(1 - y, targets - y)))
		pre_deltac = np.divide(np.square(y - targets), 2)
		if self.layers[i].activation_function == "sigmoid":
			deltac = self.sigmoidDerivative(pre_deltac)
		elif self.layers[i].activation_function == "relu":
			deltac = self.reluDerivative(pre_deltac)
		elif self.layers[i].activation_function == "leakyRelu":
			deltac = self.leakyReluDerivative(pre_deltac)
		deltaw = np.dot(deltac, self.layers[i].activations.T)
		deltab = deltac
		new_weights = self.layers[i - 1].weights_for_layer - self.learning_rate * deltaw
		new_biases = self.layers[i - 1].biases_for_layer - self.learning_rate * deltab
		for i in range(i - 1, 0, -1):
			y = self.layers[i].activations
			pre_deltac = np.divide(np.square(y - targets), 2)
			if self.layers[i].activation_function == "sigmoid":
				deltac = self.sigmoidDerivative(pre_deltac)
			elif self.layers[i].activation_function == "relu":
				deltac = self.reluDerivative(pre_deltac)
			elif self.layers[i].activation_function == "leakyRelu":
				deltac = self.leakyReluDerivative(pre_deltac)
			deltaw = np.dot(deltac, self.layers[i].activations.T)
			deltab = deltac
			self.layers[i].weights_for_layer = new_weights
			new_weights = self.layers[i - 1].weights_for_layer - self.learning_rate * deltaw
			new_biases = self.layers[i - 1].biases_for_layer - self.learning_rate * deltab
		self.layers[0].weights_for_layer = new_weights
		self.layers[0].biases_for_layer = new_biases

	def predict(self, filename, input):
		dill.load_session(filename)
		self.batch_size = 1
		self.forward_pass(input)
		a = self.layers[self.num_layers - 1].activations
		a[np.where(a == np.max(a))] = 1
		a[np.where(a != np.max(a))] = 0
		return a

	def check_accuracy(self, filename, inputs, labels):
		dill.load_session(filename)
		self.batch_size = len(inputs)
		self.forward_pass(inputs)
		a = self.layers[self.num_layers - 1].activations
		a[np.where(a == np.max(a))] = 1
		a[np.where(a != np.max(a))] = 0
		total = 0
		correct = 0
		for i in range(len(a)):
			total = total + 1
			# print(self.layers[self.num_layers - 1].activations[i], a[i], labels[i])
			print(a[i], labels[i])
			if np.equal(a[i], labels[i]).all():
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

	def sigmoidDerivative(x):
		return x * (1.0 - x)


class Layer:
	def __init__(self, activation_rows, activation_columns, weight_rows, weight_columns, activation_function):
		self.activation_rows = activation_rows
		self.activation_columns = activation_columns
		self.weight_rows = weight_rows
		self.weight_columns = weight_columns
		self.activation_function = activation_function
		self.activations = np.zeros(shape=(activation_rows, activation_columns))
		self.weights_for_layer = np.random.normal(0, 0.001, size=(weight_rows, weight_columns))
		self.biases_for_layer = np.zeros(shape=(1, weight_columns))


def __str__(self):
	return 'NodeNum: ' + str(self.num_nodes_in_layer) + '\nActivationFunction: ' + str(self.activation_function) \
		   + '\nActivations Size: ' + str(self.activations.shape) + '\nWeights Size: ' \
		   + str(self.weights_for_layer.shape) \
		   + '\nActivations: ' + str(self.activations) + '\nWeights: ' + str(self.weights_for_layer) \
		   + str(self.biases_for_layer.shape) \
		   + '\nActivations: ' + str(self.activations) + '\nBiases: ' + str(self.biases_for_layer)
