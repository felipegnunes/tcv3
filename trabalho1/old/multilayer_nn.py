import numpy as np

class Layer:
	def __init__(self, num_inputs, num_neurons):
		self.num_inputs = num_inputs
		self.num_neurons = num_neurons
		
		self.W = np.random.uniform(-10, 10, (num_neurons, 1 + num_inputs))#np.ones((num_neurons, 1 + num_inputs))
		
	def output(self, X):
		X = np.insert(X, 0, 1, axis=1)
		return self._sigmoid(X.dot(self.W.T))
	
	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
	
class NeuralNetwork:

	def __init__(self, input_size, output_size, num_hidden_layers, layer_size):
		self.input_size = input_size
		self.num_hidden_layers = num_hidden_layers
		self.layer_size = layer_size
		
		self.layers = []
		
		
		if num_hidden_layers > 0:
			first_layer = Layer(num_inputs = input_size, num_neurons = layer_size)
			self.layers.append(first_layer)
			
			for i in range(num_hidden_layers - 1):
				self.layers.append(Layer(num_inputs = layer_size, num_neurons = layer_size))
			
			output_layer = Layer(num_inputs = layer_size, num_neurons = output_size)
			self.layers.append(output_layer)
		else:
			self.layers.append(Layer(num_inputs = input_size, num_neurons = output_size))		
		
	def feed_forward(self, X):
		layer_input = X
		for layer in self.layers:
			output = layer.output(layer_input)
			layer_input = output
			
		return output
	
	def back_propagation(self, X):
		pass
			
	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
	
	def _sigmoid_derivate(self, x):
		self._sigmoid(x) * (1 - self._sigmoid(x))
		
	def train(self, X, y):
		pass
		
	def predict(self, X):
		output = self.feed_forward(X)
		return [np.argmax(y_hat) for y_hat in output]
					

def main():
	model = NeuralNetwork(input_size = 3, output_size = 3, num_hidden_layers = 1, layer_size = 3)
	print(model.layers[0].W)
	print(model.layers[1].W)
	
	X2 = np.random.uniform(-1, 1, (10, 3))

	print(model.feed_forward(X2))
	print(model.predict(X2))
	
	
if __name__ == '__main__':
	main()
