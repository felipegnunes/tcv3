#from __future__ import absolute_import
import numpy as np
import random
import sys
import time
import pickle

import dataset_loader
import logistic_regression

def main():
	# PARAMETERS
	dataset_dir = sys.argv[1]
	train_rate = float(sys.argv[2])
	
	# LOADING IMAGE NAMES AND LABELS
	
	train_set, test_set = dataset_loader.load_dataset(dataset_dir)
	print('Train set size: {}\nTest set size: {}'.format(len(train_set), len(test_set)))
	
	# SHUFFLING IMAGES
		
	random.shuffle(train_set)
	
	# LOADING IMAGES ON A NUMPY ARRAY
	
	hidden_dataset = dataset_loader.load_images(test_set)
	X = dataset_loader.load_images([row[0] for row in train_set])
	y = np.array([row[1] for row in train_set], dtype = np.int64)
	X /= 255
	
	# TRANSFORMING X IN A ARRAY
	
	hidden_dataset = hidden_dataset.reshape(hidden_dataset.shape[0], -1)
	hidden_dataset /= 255
	
	num_images = X.shape[0]
	X = X.reshape(num_images, -1)
	y = np.array([[1 if i == classification else 0 for i in range(10)] for classification in y])	
	
	# TRAIN/TEST SPLIT
	
	X_train , X_validation, y_train, y_validation = dataset_loader.train_test_split(X, y, train_rate)
	
	print('X_train: {}\nX_validation: {}\ny_train: {}\ny_validation: {}'.format(len(X_train), len(X_validation), len(y_train), len(y_validation)))		
	print(X.shape)
	
	model = logistic_regression.LogisticRegression(X.shape[1], y.shape[1], eta = 1e-2)
	model.fit(X_train, y_train, num_iterations = None, time_limit = 20, verbose = True)
	
	print('Validation Set Accuracy: {}'.format(model.accuracy(X_validation, y_validation)))
		
	print(model.predict(hidden_dataset[:10]))	
if __name__ == '__main__':
	main()
