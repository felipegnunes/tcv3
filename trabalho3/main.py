#from __future__ import absolute_import
import numpy as np
import random
import sys
import time
import copy
import os
import tensorflow as tf
import itertools

import dataset_manip		
from model import Model
					
def store_predictions(predictions, dataset_directory):
	_, test_images = dataset_manip.load_paths(dataset_directory)
	
	result = list(zip(test_images, predictions))

	result.sort(key = lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
	print('Number of predictions: {}'.format(len(result)))
	
	with open('trabalho3.txt', 'w') as f:
		for image_name, prediction in result:
			f.write('{} {}\n'.format(os.path.basename(image_name), prediction))
			
def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	DATASET_DIRECTORY = '/home/felipe/tcv3/data_part1'
	
	X, y, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
	num_classes = len(set(y))
	
	print('X.shape = ' + str(X.shape))
	print('X_hidden.shape = ' + str(X_hidden.shape))
	
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = 0.8)

	model = Model(image_shape = X.shape[1 : ], num_classes = num_classes, model_path = './model_files/model', batch_size = 250, first_run = True)
	model.train(X_train, y_train, X_validation, y_validation, 1)
	print(model.measure_accuracy(X_validation, y_validation))
	
	# ENSEMBLE
	#X, X_test, y, y_test = dataset_manip.split_dataset(X, y, rate = 0.8)
	#ensemble = Ensemble(input_shape = X.shape[1: ], num_classes = num_classes, num_models = 5, batch_size = 250)
	#print(ensemble.measure_accuracy(X_test, y_test))
	#ensemble.train(X = X, y = y, epochs_per_model = 50, train_rate = 0.8)
	#print(ensemble.measure_accuracy(X_test, y_test))
	#for i, model in enumerate(ensemble.models):
	#	print('{} on model {}'.format(model.measure_accuracy(X_test, y_test) ,i))

	predictions = model.predict(X_hidden)
	store_predictions(predictions, DATASET_DIRECTORY)
	
if __name__ == '__main__':
	main()
