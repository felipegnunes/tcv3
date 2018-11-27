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
	
def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	DATASET_DIRECTORY = '../data_part1'
	
	X, y, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
	num_classes = len(set(y))
	
	print('X.shape = ' + str(X.shape))
	print('X_hidden.shape = ' + str(X_hidden.shape))
	
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = 0.9)
	
	model = Model(image_shape = X.shape[1 : ], num_classes = num_classes, model_path = './model_files/model', batch_size = 128, first_run = True) # 1250	
	
	model.train(X_train, y_train, X_validation, y_validation, 10)
	
	
	return
	for epoch in range(10):
		#perturbate_randomly(images, horizontal_shift_range, vertical_shift_range, angle_range, contrast_alpha_range, zoom_factor_range):
		#X_train_aug = image_manip.perturbate_randomly(X_train, (-10, 10), (-10, 10), (-10, 10), (.8, 1.2), (.8, 1.2))
		model.train(X_train_aug, y_train, X_validation, y_validation, 20)
	
	print(model.measure_accuracy(X_validation, y_validation))
	
	model.train_unsupervised(X_hidden, X_validation, y_validation, 100)
	
	print(model.measure_accuracy(X_validation, y_validation))
	dataset_manip.store_predictions(dataset_manip.get_filenames('../data_part1/test'), model.predict(X_hidden), './predictions2.txt')
	
if __name__ == '__main__':
	main()
