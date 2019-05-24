import numpy as np
import dill


class ScratchNN3:

	def __init__(self, sample_num, num_nodes_list, activation_function, cost_function):
		self.sample_num = sample_num
		self.num_layers = len(num_nodes_list)
		self.num_nodes_list = num_nodes_list
		self.activation_function = activation_function
		self.cost_function = cost_function
		self.layers = []
		self.error = 0

		for i in range(self.num_layers):
			if i != self.num_layers - 1:
				layer_i = Layer(i, sample_num, num_nodes_list, activation_function[i])
			else:
				layer_i = Layer(i, sample_num, num_nodes_list, activation_function[i])
			self.layers.append(layer_i)

	def train(self, inputs, targets, num_epochs, learning_rate, filename):
		self.learning_rate = learning_rate
		for i in range(num_epochs):
			print("== EPOCH: ", i, " ==")
			# self.error = 0
			self.forward_pass(inputs)
			self.calculate_error(targets)
			self.back_pass(targets)
			print("Error: ", self.error)
		dill.dump_session(filename)

	def forward_pass(self, inputs):
		self.layers[0].activations = inputs
		for i in range(self.num_layers - 1):
			self.layers[i].zed = np.add(np.dot(self.layers[i].activations, self.layers[i].weights), self.layers[i].biases)
			self.layers[i + 1].activations = self.functionPicker(self.layers[i].zed, self.layers[i + 1].activation_function)

	def calculate_error(self, targets):
		if len(targets[0]) != self.layers[self.num_layers - 1].num_nodes_in_layer:
			print("Error: Label is not of the same shape as output layer.")
			print("Label: ", len(targets), " : ", len(targets[0]))
			print("Out: ", len(self.layers[self.num_layers - 1].activations), " : ", len(self.layers[self.num_layers - 1].activations[0]))
			return

		if self.cost_function == "mean_squared":
			self.error = np.mean(np.divide(np.square(np.subtract(targets, self.layers[self.num_layers - 1].activations)), 2))
		elif self.cost_function == "cross_entropy":
			self.error = np.negative(np.sum(np.multiply(targets, np.log(self.layers[self.num_layers - 1].activations))))

	def back_pass(self, targets):
		for i in range(self.num_layers - 1, 0, -1):
			if i == self.num_layers - 1:
				self.layers[i].activations_delta = np.divide(self.layers[i].activations - targets, targets.shape[0])
				# self.layers[i].weights_delta = np.dot(self.layers[i - 1].activations.T, self.layers[i].activations_delta)
				# self.layers[i].biases_delta = np.sum(self.layers[i].activations_delta, axis=0, keepdims=True)
				self.layers[i].weights -= np.multiply(self.learning_rate, np.dot(self.layers[i - 1].activations.T, self.layers[i].activations_delta))
				self.layers[i].biases -= np.multiply(self.learning_rate, np.sum(self.layers[i].activations_delta, axis=0, keepdims=True))
			else:
				self.layers[i].zed_delta = np.dot(self.layers[i + 1].activations_delta, self.layers[i + 1].weights.T)
				self.layers[i].activations_delta = np.multiply(self.layers[i].zed_delta, self.functionDerivativePicker(self.layers[i].activations, self.layers[i].activation_function))
				# self.layers[i].weights_delta = np.dot(self.layers[i - 1].activations.T, self.layers[i].activations_delta)
				# self.layers[i].biases_delta = np.sum(self.layers[i].activations_delta, axis=0)
				self.layers[i].weights -= np.multiply(self.learning_rate, np.dot(self.layers[i - 1].activations.T, self.layers[i].activations_delta))
				self.layers[i].biases -= np.multiply(self.learning_rate, np.sum(self.layers[i].activations_delta, axis=0))

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

	def sigmoid(self, layer):
		return np.divide(1, np.add(1, np.exp(layer)))

	def sigmoidDerivative(self, x):
		return x * (1.0 - x)

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

	def functionPicker(self, input, function):
		if function == "sigmoid":
			return self.sigmoid(input)
		elif function == "relu":
			return self.relu(input)
		elif function == "leakyRelu":
			return self.leakyRelu(input)
		else:
			print('Malformed function declaration.')
			return

	def functionDerivativePicker(self, input, function):
		if function == "sigmoidDerivative":
			return self.sigmoidDerivative(input)
		elif function == "reluDerivative":
			return self.reluDerivative(input)
		elif function == "leakyReluDerivative":
			return self.leakyReluDerivative(input)
		else:
			print('Malformed function declaration.')
			return


class Layer:
	def __init__(self, layer_id, sample_num, num_nodes_list, activation_function):
		self.layer_id = layer_id
		self.sample_num = sample_num
		self.num_nodes_in_layer = num_nodes_list[layer_id]
		self.num_nodes_in_next_layer = num_nodes_list[layer_id]
		self.activation_function = activation_function

		if layer_id == 0:
			self.zed = np.zeros(shape=(1, 1))
			self.zed_delta = np.zeros(shape=(1, 1))
			self.activations = np.zeros(shape=(sample_num, num_nodes_list[layer_id]))
			self.activations_delta = np.zeros(shape=(sample_num, num_nodes_list[layer_id]))
			self.weights = np.zeros(shape=(1, 1))
			self.weights_delta = np.zeros(shape=(1, 1))
			self.biases = np.zeros(shape=(1, 1))
			self.biases_delta = np.zeros(shape=(1, 1))
		else:
			self.zed = np.zeros(shape=(sample_num, num_nodes_list[layer_id]))
			self.zed_delta = np.zeros(shape=(sample_num, num_nodes_list[layer_id]))
			self.activations = np.zeros(shape=(sample_num, num_nodes_list[layer_id]))
			self.activations_delta = np.zeros(shape=(sample_num, num_nodes_list[layer_id]))
			self.weights = np.random.normal(0, 0.001, size=(num_nodes_list[layer_id - 1], num_nodes_list[layer_id]))
			self.weights_delta = np.random.normal(0, 0.001, size=(num_nodes_list[layer_id - 1], num_nodes_list[layer_id]))
			self.biases = np.zeros(shape=(1, num_nodes_list[layer_id - 1]))
			self.biases_delta = np.zeros(shape=(1, num_nodes_list[layer_id - 1]))


def __str__(self):
	return 'NodeNum: ' + str(self.num_nodes_in_layer) + '\nActivationFunction: ' + str(self.activation_function) \
		   + '\nActivations Size: ' + str(self.activations.shape) + '\nWeights Size: ' \
		   + str(self.weights.shape) \
		   + '\nActivations: ' + str(self.activations) + '\nWeights: ' + str(self.weights) \
		   + str(self.biases.shape) \
		   + '\nActivations: ' + str(self.activations) + '\nBiases: ' + str(self.biases)
