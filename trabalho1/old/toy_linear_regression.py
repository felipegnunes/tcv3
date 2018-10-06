import numpy as np
import matplotlib.pyplot as plt
import random
import time

def mean_squared_error(m, b, X, y):
	return np.mean((y - (m * X + b)) ** 2)
	
def gradient_step(m, b, X, y, learning_rate):
	m_gradient = 0
	b_gradient = 0
	num_points = len(X)
	
	for i in range(num_points):
		m_gradient += (-2 * X[i] * (y[i] - (m * X[i] + b)))/num_points
		b_gradient += (-2 * (y[i] - (m * X[i] + b)))/num_points
	
	new_m = m - (learning_rate * m_gradient)
	new_b = b - (learning_rate * b_gradient)
	
	return new_m, new_b		
	
def linear_regression(X, y, m = 0, b = 0, iterations = 1000, learning_rate = 0.001):
	N = len(X)
	
	for i in range(iterations):
		y_hat = m*X + b
		
		m_gradient = -2 * np.mean(np.sum( X * (y - y_hat) ))
		b_gradient = -2 * np.mean(np.sum( y - y_hat ))
		
		m -= learning_rate * m_gradient
		b -= learning_rate * b_gradient
	
	error = np.mean(np.power(y - (m*X + b), 2))
	
	return m, b, error
	
def _linear_regression(X, y, m, b, iterations, learning_rate):
	N = X.shape[0]
	
	for i in range(iterations):
		y_hat = [np.dot(m, x) + b for x in X]
		
		for j in range(len(m)):
			m_gradient = -2 * np.mean(np.sum( X * (y - y_hat) ))
		
		b_gradient = -2 * np.mean(np.sum( y - y_hat ))
	
	error = np.mean		

def main():
	sample_size = int(5000)
	
	f = lambda x: 5.7 * x + 10.231

	X = np.empty([sample_size], dtype=np.float64)
	y = np.empty([sample_size], dtype=np.float64)

	for i in range(sample_size):
		x = random.uniform(0.0, 1e2)
		X[i] = x
		y[i] = f(x)
		
	print(max(X), max(y))	
	st = time.time()
	print(linear_regression(X, y, m = 0, b = 0, iterations = int(1e3), learning_rate = 1e-8))
	fn = time.time()
	
	time_seconds = fn - st
	time_minutes = time_seconds/60
	
	print('Wall clock time: {} s.'.format(time_seconds))
	print('Wall clock time: {} min.'.format(time_minutes))
	
	#print('MSE: {}'.format(mean_squared_error(m, b, X, y)))
	
	#real_m = np.random.rand(5)
	#real_b = np.random.rand(1)
	#print(real_m)
	
	#X = np.random.rand(sample_size, 5)#np.empty([sample_size, 5], dtype = np.float64)
	#y = np.empty([sample_size], dtype = np.float64)
	#for i in range(sample_size):
	#	y[i] = np.dot(real_m, X[i]) + real_b #np.empty([sample_size], dtype = np.float64)
	#print(X)
	#print(y)
	#for i in range(sample_size):
	#	x = random.uniform(0.0, 1e2)
	#	X[i] = x
	#	y[i] = f(x)
	
	#m = np.zeros([5], dtype = np.float64)
	#b = np.zeros([1], dtype = np.float64)
	
	#for i in range(50):
	#	m, b = gradient_step(m, b, X, y, 0.001)
	#	#print(m, b)
		
	#print(m, b)
		

if __name__ == '__main__':
	main()
