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

def load_images(dataset):
	example_image_path, _ = dataset[0]
	example_image = cv2.imread(example_image_path, cv2.IMREAD_UNCHANGED)
	
	if (len(example_image.shape) < 3):
		height, width = example_image.shape
		num_channels = 1
	else:
		height, width, num_channels = example_image.shape
	
	num_images = len(dataset)
	
	X = np.empty([num_images, height, width, num_channels], dtype = np.float64)
	y = np.empty([num_images], dtype = np.int64)
	
	i = 0
	for image_path, label in dataset:	
		X[i] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).reshape(height, width, num_channels)
		y[i] = label
		i += 1
	
	return X, y
	
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
	
	X, y = load_images(train_set)
	X /= 255
	
	print(min(X), min(y))
	print(max(X), max(y))
	
	print('X: {}\ny: {}'.format(X.shape, y.shape))	
		
	t = random.randint(0, len(X))
	print(t)	
	cv2.imshow('x_train', X[t])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
if __name__ == '__main__':
	main()
