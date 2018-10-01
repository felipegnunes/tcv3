import numpy as np

class LogisticRegression:

	def __init__(self, input_size, output_size, eta = 0.1):
		self.eta = eta
		self.W = np.random.uniform(-0.01, 0.01, (input_size, output_size))
		self.B = np.random.uniform(-0.01, 0.01, (1, output_size))
	
	def fit(self, X, y, num_iterations):
		m = X.shape[0]
		for _ in range(num_iterations):
			y_hat = self.output(X)
			error = y_hat - y
			self.W -= self.eta * 2 * 1/m * error.T.dot(X).T
			self.B -= self.eta * 2 * 1/m * error.sum(axis = 0)
	
	def output(self, X):
		return X.dot(self.W) + self.B
	
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

class SigLogisticRegression:

	def __init__(self, input_size, output_size, eta = 0.1):
		self.input_size = input_size
		self.output_size = output_size
		self.eta = eta
		self.W = np.random.uniform(-0.01, 0.01, (input_size, output_size))
		self.B = np.random.uniform(-0.01, 0.01, (1, output_size))
	
	def fit(self, X, y, num_iterations):
		m = X.shape[0]
		for _ in range(num_iterations):
			y_hat = self.output(X)
			error = y_hat - y
			
			for i in range(m):
				u = self.eta * 2 * 1/m * (np.multiply(np.multiply(error[i], y_hat[i]), (1 - y_hat[i]))).T
				for j in range(self.input_size):
					self.W -= u.dot(X[i, j]).T
				self.B -= u
	
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
	
	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))	
		
def main():
	np.random.seed(0)
	
	batch_size = 1000
	input_size = 5
	output_size = 10
	eta = 1e-2
	sigmoid = lambda x: 1 / (1 + np.exp(-x))
	
	X = np.random.uniform(-10, 10, (batch_size, input_size))
	real_W = np.random.uniform(-10, 10, (input_size, output_size))
	real_B = np.random.uniform(-10, 10, (1, output_size))		
	y = sigmoid(X.dot(real_W) + real_B)
	y_ = y.argmax(axis = 1)
	print(y_)
	
	
	model = SigLogisticRegression(input_size, output_size, eta)
	model.fit(X, y, 1000)
	
	print(model.W - real_W)
	print()
	print(model.B - real_B)
	
	#print(model.predict(X))
	#print(y.argmax(axis = 1))
	print(model.accuracy(X, y))
	print(model.mean_squared_error(X, y))
		
if __name__ == '__main__':
	main()
