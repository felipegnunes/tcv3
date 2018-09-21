#from __future__ import absolute_import
import numpy as np
import random
import sys
import time

import dataset_loader
import linear_regression
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
	
	X = dataset_loader.load_images([row[0] for row in train_set])
	y = np.array([row[1] for row in train_set], dtype = np.int64)
	X /= 255

	# TRANSFORMING X IN A ARRAY
	
	num_images = X.shape[0]
	X = X.reshape(num_images, -1)	
	
	# TRAIN/TEST SPLIT
	
	X_train , X_validation, y_train, y_validation = dataset_loader.train_test_split(X, y, train_rate)
	
	print('X_train: {}\nX_validation: {}\ny_train: {}\ny_validation: {}'.format(len(X_train), len(X_validation), len(y_train), len(y_validation)))		
		
	#model = linear_regression.LinearRegression(eta = 1e-2, n_iter = 1000).fit(X_train, y_train) 
	#model = linear_regression.LinearRegressionSGD(eta = 1e-3, n_iter = 100000).fit(X_train, y_train)
	
	#print(model.predict(X_validation[:10]))
	#print(y_validation[:10])
	#print(model.mean_squared_error(X_validation, y_validation))
	
	start = time.time()
	model = logistic_regression.LogisticRegression(eta = 1e-1, n_iter = 20000).fit(X_train, y_train)
	finish = time.time()
	
	time_seconds = finish - start
	time_minutes = time_seconds/60
	
	print('Wall clock time: {:.2f} s.'.format(time_seconds))
	print('Wall clock time: {:.2f} min.'.format(time_minutes))
	
	print(model.score(X_train, y_train))
	print(model.score(X_validation, y_validation))
		
if __name__ == '__main__':
	main()
