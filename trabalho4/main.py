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
from model import Model, Ensemble
					
def store_predictions(predictions, dataset_directory):
	_, test_images = dataset_manip.load_paths(dataset_directory)
	
	result = list(zip(test_images, predictions))

	result.sort(key = lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
	print('Number of predictions: {}'.format(len(result)))
	
	with open('predictions.txt', 'w') as f:
		for image_name, prediction in result:
			f.write('{} {}\n'.format(os.path.basename(image_name), prediction))
			
def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	DATASET_DIRECTORY = '../data_part1'
	
	X, y, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
	num_classes = len(set(y))
	
	print('X.shape = ' + str(X.shape))
	print('X_hidden.shape = ' + str(X_hidden.shape))
	
	#X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = 0.8)
	#model = Model(image_shape = X.shape[1 : ], num_classes = num_classes, model_path = './model_files/model', batch_size = 1250, first_run = True)
	#model.train(X_train, y_train, X_validation, y_validation, 1)
	#print(model.measure_accuracy(X_validation, y_validation))
	#print(model.predict(X_hidden))
	
	# ENSEMBLE
	
	ensemble = Ensemble(input_shape = X.shape[1: ], num_classes = num_classes, num_models = 2, batch_size = 1250)
	ensemble.train(X = X, y = y, epochs_per_model = 1, split_rate = 0.8)
	
	predictions = ensemble.predict(X_hidden)
	#print(predictions)
	#store_predictions(predictions, DATASET_DIRECTORY)
	
if __name__ == '__main__':
	main()
