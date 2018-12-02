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
	
	ens = Ensemble(input_shape = (77, 71, 1), num_classes = 10, num_models = 11, batch_size = 512, path = './ensemble_files', load = False)
	ens.train(X = X, y = y, epochs_per_model = 300, split_rate = 0.9)
	print(ens.measure_accuracy(X, y))
	
	return
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = 0.5)
	
	model = Model(image_shape = X.shape[1 : ], num_classes = num_classes, model_path = './model_files/model', batch_size = 512, first_run = True) # 1250	
	
	model.train(X_train, y_train, X_validation, y_validation, 500)
	model.train_unsupervised(X_hidden, X_validation, y_validation, 200)
	
	print('Final Accuracy: {}'.format(model.measure_accuracy(X_validation, y_validation)))
	#dataset_manip.store_predictions(dataset_manip.get_filenames('../data_part1/test'), model.predict(X_hidden), './predictions2.txt')
	
if __name__ == '__main__':
	main()
