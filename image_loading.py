from __future__ import absolute_import
import os
import numpy as np
import cv2
import random
import sys

def get_immediate_subdirectories(directory_path):
	return [sub_directory for sub_directory in os.listdir(directory_path) 
		if os.path.isdir(os.path.join(directory_path, sub_directory))]

def get_filenames(files_path):
	filenames = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))]
	return filenames

def load_filenames(dataset_directory):
	train_dir_path = os.path.join(dataset_directory, 'train')
	test_dir_path = os.path.join(dataset_directory, 'test')
	
	train_directories = get_immediate_subdirectories(train_dir_path)
	test_directories = get_immediate_subdirectories(test_dir_path)
	
	train_images = []
	
	for directory in train_directories:
		directory_path = os.path.join(train_dir_path, directory)
		label = int(directory)
		image_list = get_filenames(directory_path)
	
		for image in image_list:
			image_path = os.path.join(directory_path, image)
			train_images.append((image_path, label))
	
	test_images = get_filenames(test_dir_path)
	
	return train_images, test_images		

def load_images(dataset):
	example_img_path = dataset[0][0]
	example_image = cv2.imread(example_img_path, cv2.IMREAD_UNCHANGED)
	
	height = example_image.shape[0] 
	width = example_image.shape[1]
	if (len(example_image.shape) < 3):
		num_channels = 1
	else:
		num_channels = example_image.shape[2]
	
	num_imgs = len(dataset)
	
	X = np.empty([num_imgs, height, width, num_channels], dtype = np.uint8)
	y = np.empty([num_imgs], dtype = np.int64)
	
	i = 0
	for image_path, label in dataset:	
		X[i] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).reshape(height, width, num_channels)
		y[i] = label
		i += 1
	
	return X, y
	
def main():
	# PARAMETERS
	dataset_dir = sys.argv[1]
	train_test_rate = float(sys.argv[2])
	
	# LOADING IMAGE NAMES AND LABELS
	
	train_set, validation_set = load_filenames(dataset_dir)
	print(len(train_set))
	print(len(validation_set))
	
	# SHUFFLING IMAGES
		
	random.shuffle(train_set)
	
	# LOADING IMAGES ON A NUMPY ARRAY
	
	X, y = load_images(train_set)
	
	#sample_img_path = imgs[0][0]
	#example_image = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
	
	#if (len(example_image.shape) < 3):
	#	height, width = example_image.shape
	#	num_channels = 1
	#else:
	#	height, width, num_channels = example_image.shape
			
	#split_point = int(train_test_rate * len(imgs))
	#num_train_imgs = split_point
	#num_test_imgs = len(imgs) - split_point
	
	#X_train = np.empty([num_train_imgs, height, width, num_channels], dtype=np.uint8)
	#X_test = np.empty([num_test_imgs, height, width, num_channels], dtype=np.uint8)
	#y_train = np.empty([num_train_imgs], dtype=np.int64)
	#y_test = np.empty([num_test_imgs], dtype=np.int64)
	
	#i = 0
	#for image_path, label in imgs[ : split_point]:	
	#	X_train[i] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).reshape(height, width, num_channels)
	#	y_train[i] = label
	#	i += 1
	
	#i = 0
	#for image_path, label in imgs[split_point : ]:
	#	X_test[i] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).reshape(height, width, num_channels)
	#	y_test[i] = label
	#	i += 1
		
	print('X: {}\ny: {}'.format(X.shape, y.shape))	
		
	cv2.imshow('x_train', X[0])
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
	
if __name__ == '__main__':
	main()
