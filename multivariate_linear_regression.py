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
			self.w += (self.eta * errors.dot(X))/m
			
		return self
	
	def predict(self, X):
		return np.insert(X, 0, 1, axis = 1).dot(self.w)
	
	def score(self, X, y):
		return 1.0 - sum((self.predict(X) - y) ** 2) / sum((y - np.mean(y)) ** 2)

def main():
	num_samples = 500
	sample_dimension = 5
	
	X = np.random.rand(num_samples, sample_dimension)
	real_w = np.array([5, 8, 3, 4, 7]) #np.random.rand(sample_dimension)
	real_b = 7.91 #np.random.rand()
	y = [np.dot(x, real_w) + real_b for x in X]
	
	#print(X)
	#print(y)
	
	model = LinearRegression(eta = 1e-2, n_iter = int(1e5)).fit(X, y)
	print(model.score(X, y))
	print(model.w)
	
if __name__ == '__main__':
	main()
