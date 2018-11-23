import os
import cv2
import numpy as np

import autoencoder
import dataset_manip
import image_manip

def resize_image_set(images, new_size):
	num_samples = images.shape[0]
	resized_images = np.empty(shape = (num_samples, ) + new_size, dtype = np.float32)
	
	for i in range(num_samples):
		resized_images[i] = cv2.resize(images[i], new_size[ : 2], interpolation = cv2.INTER_AREA).reshape(new_size)
	
	return resized_images

def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	DATASET_DIRECTORY = '../data_part1'
	
	X, y, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
	
	X = resize_image_set(X, (64, 64, 1))
	X_hidden = resize_image_set(X_hidden, (64, 64, 1))
	
	print('X.shape = ' + str(X.shape))
	print('X_hidden.shape = ' + str(X_hidden.shape))
	
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = 0.9)
	
	model = autoencoder.Autoencoder(input_shape = X.shape[1 : ], model_path = './model_files/autoencoder_model', batch_size = 8, first_run = True)
	print(model.measure_loss(X_validation))
	model.train(X_train, X_validation, num_epochs = 100)
	print(model.measure_loss(X_validation))
if __name__ == '__main__':
	main()

