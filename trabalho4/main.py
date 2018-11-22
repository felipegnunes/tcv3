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
import image_manip
from model import Model
from ensemble import Ensemble
					
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
	
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = 0.5)
	
	model = Model(image_shape = X.shape[1 : ], num_classes = num_classes, model_path = './model_files/model', batch_size = 100, first_run = True)
	
	model.train(X_train, y_train, X_validation, y_validation, 20)
	for epoch in range(10):
		#perturbate_randomly(images, horizontal_shift_range, vertical_shift_range, angle_range, contrast_alpha_range, zoom_factor_range):
		X_train_aug = image_manip.perturbate_randomly(X_train, (-10, 10), (-10, 10), (-10, 10), (.8, 1.2), (.8, 1.2))
		model.train(X_train_aug, y_train, X_validation, y_validation, 20)
	
	for epoch in range(10):
		#perturbate_randomly(images, horizontal_shift_range, vertical_shift_range, angle_range, contrast_alpha_range, zoom_factor_range):
		X_train_aug = image_manip.perturbate_randomly(X_train, (-15, 15), (-15, 15), (-15, 15), (.7, 1.3), (.7, 1.3))
		model.train(X_train_aug, y_train, X_validation, y_validation, 10)
		
	for epoch in range(10):
		#perturbate_randomly(images, horizontal_shift_range, vertical_shift_range, angle_range, contrast_alpha_range, zoom_factor_range):
		X_train_aug = image_manip.perturbate_randomly(X_train, (-20, 20), (-20, 20), (-20, 20), (.5, 1.5), (.8, 1.2))
		model.train(X_train_aug, y_train, X_validation, y_validation, 10)
		
	print(model.measure_accuracy(X_validation, y_validation))
	#print(model.predict(X_hidden))
	
if __name__ == '__main__':
	main()
