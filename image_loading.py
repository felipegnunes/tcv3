#from __future__ import absolute_import
import os
import numpy as np
import cv2
import random
import sys

def get_immediate_subdirectories(directory_path):
	return [subdirectory for subdirectory in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, subdirectory))]

def get_filenames(directory_path):
	return [filename for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]

def load_dataset(dataset_directory):
	train_directory_path = os.path.join(dataset_directory, 'train')
	test_directory_path = os.path.join(dataset_directory, 'test')
	
	labeled_directories = get_immediate_subdirectories(train_directory_path)
	
	train_images = []
	
	for directory in labeled_directories:
		directory_path = os.path.join(train_directory_path, directory)
		label = int(directory)
		image_list = get_filenames(directory_path)
	
		for image_name in image_list:
			image_path = os.path.join(directory_path, image_name)
			train_images.append((image_path, label))
	
	test_images = get_filenames(test_directory_path)
	
	return train_images, test_images		

#def load_images(dataset):
#	example_image_path, _ = dataset[0]
#	example_image = cv2.imread(example_image_path, cv2.IMREAD_UNCHANGED)
#	
#	if (len(example_image.shape) < 3):
#		height, width = example_image.shape
#		num_channels = 1
#	else:
#		height, width, num_channels = example_image.shape
#	
#	num_images = len(dataset)
#	
#	X = np.empty([num_images, height, width, num_channels], dtype = np.float64)
#	y = np.empty([num_images], dtype = np.int64)
#	
#	i = 0
#	for image_path, label in dataset:
#		X[i] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).reshape(height, width, num_channels)
#		y[i] = label
#		i += 1
#	
#	return X, y

def load_images(image_list):
	example_image_path = image_list[0]
	example_image = cv2.imread(example_image_path, cv2.IMREAD_UNCHANGED)
	
	if (len(example_image.shape) < 3):
		height, width = example_image.shape
		num_channels = 1
	else:
		height, width, num_channels = example_image.shape
	
	num_images = len(image_list)
	
	images = np.empty([num_images, height, width, num_channels], dtype = np.float64)
	
	i = 0
	for image_path in image_list:
		images[i] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).reshape(height, width, num_channels)
		i += 1
	
	return images
	
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
		X = np.insert(X, 0, 1, axis=1)
		self.w = np.ones(X.shape[1])

		for _ in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)
				
			for x, target in zip(X, y):
				output = x.dot(self.w)
				error = target - output
				self.w += self.eta * error * x

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
	
def main():
	# PARAMETERS
	dataset_dir = sys.argv[1]
	train_split_rate = float(sys.argv[2])
	
	# LOADING IMAGE NAMES AND LABELS
	
	train_set, test_set = load_dataset(dataset_dir)
	print('Train set size: {}\nTest set size: {}'.format(len(train_set), len(test_set)))
	
	# SHUFFLING IMAGES
		
	random.shuffle(train_set)
	
	# LOADING IMAGES ON A NUMPY ARRAY
	
	X = load_images([row[0] for row in train_set])
	y = np.array([row[1] for row in train_set], dtype = np.int64)
	X /= 255

	# TRANSFORMING X IN A ARRAY
	
	num_images = X.shape[0]
	X = X.reshape(num_images, -1)	
	
	# TRAIN/TEST SPLIT
	
	split_point = int(num_images * train_split_rate)
	X_train = X[ : split_point]
	X_validation = X[split_point : ]
	y_train = y[ : split_point]
	y_validation = y[split_point : ]
	
	print('X_train: {}\nX_validation: {}\ny_train: {}\ny_validation: {}'.format(len(X_train), len(X_validation), len(y_train), len(y_validation)))		
		
	model = LinearRegression(eta = 1e-2, n_iter = 1000000).fit(X_train, y_train) #LinearRegressionSGD(eta = 1e-3, n_iter = 10000).fit(X_train, y_train)
	print(model.predict(X_validation[:10]))
	print(y_validation[:10])
	print(model.mean_squared_error(X_validation, y_validation))
		
if __name__ == '__main__':
	main()
