import numpy as np
import time

class OneHiddenLayer:

	def __init__(self, input_size, output_size, hidden_layer_size, eta = 0.1):
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_layer_size = hidden_layer_size
		self.eta = eta
		
		self.W1 = np.random.uniform(-0.01, 0.01, (input_size, hidden_layer_size))
		self.W2 = np.random.uniform(-0.01, 0.01, (hidden_layer_size, output_size))
		self.B1 = np.random.uniform(-0.01, 0.01, (1, hidden_layer_size))
		self.B2 = np.random.uniform(-0.01, 0.01, (1, output_size))
		
	def fit(self, X, y, num_iterations = None, time_limit = None, verbose = False):
		m = X.shape[0]
		iteration = 0
		elapsed_time = 0
		start = time.time()
		
		while (not num_iterations or iteration <= num_iterations) and (not time_limit or elapsed_time < time_limit):
			iteration += 1
			
			O1 = self._sigmoid(X.dot(self.W1) + self.B1)
			O2 = self._sigmoid(O1.dot(self.W2) + self.B2)
			error = O2 - y
			u = self.eta * 2 * 1/m * (np.multiply(np.multiply(error, O2), (1 - O2)))
			v = np.multiply(u.dot(self.W2.T), (np.multiply(O1, 1 - O1)))
			
			self.W1 -= v.T.dot(X).T
			self.B1 -= v.sum(axis = 0)  
			self.W2 -= u.T.dot(O1).T
			self.B2 -= u.sum(axis = 0)
			
			elapsed_time = (time.time() - start)/60
			
			if verbose and iteration % 10 == 0:
				print('Iteration {}\nAccuracy: {}\nMean Squared Error: {}'.format(iteration, self.accuracy(X, y), self.mean_squared_error(X, y)))
			
			if time_limit and elapsed_time > time_limit:
				if verbose:
					print('Iteration {}\nAccuracy: {}\nMean Squared Error: {}'.format(iteration, self.accuracy(X, y), self.mean_squared_error(X, y)))
					print('Elapsed Time: {} min.\nTime per Iteration: {} s.'.format(elapsed_time, (elapsed_time*60)/iteration))
				break
			
	
	def output(self, X):
		return self._sigmoid(self._sigmoid(X.dot(self.W1) + self.B1).dot(self.W2) + self.B2)
	
	def predict(self, X):
		y_hat = self.output(X)
		return y_hat.argmax(axis = 1)	
	
	def accuracy(self, X, y):	
		predictions = self.predict(X)
		correct_predictions = y.argmax(axis = 1)
		
		return (predictions == correct_predictions).mean() 
	
	def mean_squared_error(self, X, y):
		y_hat = self.output(X)
		return ((y_hat - y) ** 2).mean()
	
	def _sigmoid(self, Z):
		return 1/(1 + np.exp(-Z))

