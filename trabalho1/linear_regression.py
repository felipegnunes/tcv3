import numpy as np

class LinearRegression:
	def __init__(self, eta = 0.1, n_iter = 1000):
		self.eta = eta
		self.n_iter = n_iter
	
	def fit(self, X, y):
		X = np.insert(X, 0, 1, axis = 1)
		self.w = np.ones(X.shape[1])		
		m = X.shape[0]
		
		for _ in range(self.n_iter):
			output = X.dot(self.w)
			errors = y - output
			if (_ % 1000 == 0):
				print(_)
				print(sum((errors) ** 2)/X.shape[0])
			self.w += (self.eta * errors.dot(X))/m
			
		return self
	
	def predict(self, X):
		return np.insert(X, 0, 1, axis = 1).dot(self.w)
	
	def score(self, X, y):
		return 1.0 - sum((self.predict(X) - y) ** 2) / sum((y - np.mean(y)) ** 2)
	
	def mean_squared_error(self, X, y):
		y_hat = self.predict(X)
		return sum((y - y_hat) ** 2)/X.shape[0]
	
class LinearRegressionSGD:
	def __init__(self, eta = 0.1, n_iter = 1000, shuffle = True):
		self.eta = eta
		self.n_iter = n_iter
		self.shuffle = shuffle

	def fit(self, X, y):
		X = np.insert(X, 0, 1, axis = 1)
		self.w = np.ones(X.shape[1])

		for _ in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)
				
			for x, target in zip(X, y):
				output = x.dot(self.w)
				error = target - output
				self.w += self.eta * error * x
				
			if (_ % 1000 == 0):
				mse = sum((y - X.dot(self.w)) ** 2)/X.shape[0]
				print('{} -> {}'.format(_, mse))

		return self


	def _shuffle(self, X, y):
		r = np.random.permutation(len(y))
		return X[r], y[r]	
	
	def predict(self, X):
		return np.insert(X, 0, 1, axis = 1).dot(self.w)
	
	def score(self, X, y):
		return 1.0 - sum((self.predict(X) - y) ** 2) / sum((y - np.mean(y)) ** 2)
	
	def mean_squared_error(self, X, y):
		y_hat = self.predict(X)
		return sum((y - y_hat) ** 2)/X.shape[0]
