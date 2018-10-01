import numpy as np
import time

class LogisticRegression:

	def __init__(self, input_size, output_size, eta = 0.1):
		self.input_size = input_size
		self.output_size = output_size
		self.eta = eta
		self.W = np.random.uniform(-0.01, 0.01, (input_size, output_size))
		self.B = np.random.uniform(-0.01, 0.01, (1, output_size))
	
	def fit(self, X, y, num_iterations = None, time_limit = None, verbose = False):
		m = X.shape[0]
		iteration = 0
		elapsed_time = 0
		start = time.time()
		
		while (not num_iterations or iteration <= num_iterations) and (not time_limit or elapsed_time < time_limit):
			iteration += 1
			
			y_hat = self.output(X)
			error = y_hat - y
			u = self.eta * 2 * 1/m * (np.multiply(np.multiply(error, y_hat), (1 - y_hat)))
			self.W -= u.T.dot(X).T
			self.B -= u.sum(axis = 0)
			elapsed_time = (time.time() - start)/60
			
			if verbose and iteration % 1000 == 0:
				print('Iteration {}\nAccuracy: {}\nMean Squared Error: {}'.format(iteration, self.accuracy(X, y), self.mean_squared_error(X, y)))
			
			if time_limit and elapsed_time > time_limit:
				if verbose:
					print('Iteration {}\nAccuracy: {}\nMean Squared Error: {}'.format(iteration, self.accuracy(X, y), self.mean_squared_error(X, y)))
					print('Elapsed Time: {} min.\nTime per Iteration: {} s.'.format(elapsed_time, (elapsed_time*60)/iteration))
				break
			
	
	def output(self, X):
		return self._sigmoid(X.dot(self.W) + self.B)
	
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
