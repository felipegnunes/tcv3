#from __future__ import absolute_import
import numpy as np
import random
import sys
import time
import copy
import os

import dataset_loader
import logistic_regression
import one_hidden_layer

def main():
	dataset_dir = sys.argv[1]
	train_rate = float(sys.argv[2])
	
	# LOADING IMAGE NAMES AND LABELS
	
	labeled_set, hidden_set = dataset_loader.load_dataset(dataset_dir)
	print('Labeled set size: {}\nHidden set size: {}'.format(len(labeled_set), len(hidden_set)))
	
	# SHUFFLING IMAGES
		
	random.shuffle(labeled_set)
	
	# LOADING IMAGES ON A NUMPY ARRAY
	
	X_hidden = dataset_loader.load_images(hidden_set)
	X = dataset_loader.load_images([row[0] for row in labeled_set])
	y = np.array([row[1] for row in labeled_set], dtype = np.int64)
	y = np.array([[1 if i == classification else 0 for i in range(10)] for classification in y])	
	
	# TRANSFORMING X IN A ARRAY
	
	X_hidden = X_hidden.reshape(X_hidden.shape[0], -1)
	X_hidden /= 255
	
	X = X.reshape(X.shape[0], -1)
	X /= 255
	
	# TRAIN/TEST SPLIT
	
	X_train , X_validation, y_train, y_validation = dataset_loader.dataset_split(X, y, train_rate)
	
	print('X_train: {}\nX_validation: {}\ny_train: {}\ny_validation: {}'.format(X_train.shape, X_validation.shape, y_train.shape, y_validation.shape))		
	
	# PREDICTION
	
	#model = logistic_regression.LogisticRegression(X.shape[1], y.shape[1], eta = 1e-1)
	#model = one_hidden_layer.OneHiddenLayer(X.shape[1], y.shape[1], 60, eta = 1e-1)
	
	#best_model = copy.deepcopy(model)
	#best_acc = model.accuracy(X_validation, y_validation)
	
	#for iteration in range(1, 61):
	#	model.fit(X_train, y_train, time_limit = 1)
		
	#	model_acc = model.accuracy(X_validation, y_validation)
	#	if model_acc > best_acc:
	#		best_acc = model_acc
	#		best_model = copy.deepcopy(model)
	#	print(best_acc)	
	
	#predictions = best_model.predict(X_hidden)
	#result = list(zip(hidden_set, predictions))
	#result.sort(key = lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
	#print('Number of predictions: {}'.format(len(result)))
	
	#with open('multilayer_perceptron.txt', 'w') as f:
	#	for image_name, prediction in result:
	#		f.write('{} {}\n'.format(os.path.basename(image_name), prediction))
	
	
	# ----------------------------------------------------------------------------------------------
				
	#model = one_hidden_layer.OneHiddenLayer(X.shape[1], y.shape[1], 60, eta = 1e-1)
	#model2 = copy.deepcopy(model)
	
	#model.fit(X_train, y_train, time_limit = 1)
	#print(model.accuracy(X_validation, y_validation))
	#print(model2.accuracy(X_validation, y_validation))
	
	
	
	#hidden_layer_sizes = [20, 30, 40, 50, 60]
	#performances = []
	
	#for size in hidden_layer_sizes:
	#	print('Size: {}'.format(size))
	#	model = one_hidden_layer.OneHiddenLayer(X.shape[1], y.shape[1], size, eta = 1e-1)	
	
	#	best_acc = 0
	#	for minute in range(1, 31):
	#		print('\tMinute {}'.format(minute))
	#		model.fit(X_train, y_train, num_iterations = None, time_limit = 1.0, verbose = False)
	#		model_acc = model.accuracy(X_validation, y_validation)
	#		if model_acc > best_acc:
	#			best_acc = model_acc
			#print(best_acc)
		
	#	performances.append(best_acc)	

	#print(list(zip(hidden_layer_sizes, performances)))
	
	#print('Validation Set Accuracy: {}'.format(model.accuracy(X_validation, y_validation)))	

if __name__ == '__main__':
	main()
