import os
import cv2
import numpy as np

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
	
	test_image_names = get_filenames(test_directory_path)
	test_images = [os.path.join(test_directory_path, image_name) for image_name in test_image_names]
	
	return train_images, test_images

def load_images(image_list):
	example_image_path = image_list[0]
	example_image = cv2.imread(example_image_path, cv2.IMREAD_UNCHANGED)
	
	if (len(example_image.shape) < 3):
		height, width = example_image.shape
		num_channels = 1
	else:
		height, width, num_channels = example_image.shape
	
	num_images = len(image_list)
	
	images = np.empty([num_images, height, width, num_channels], dtype = np.float64)
	
	i = 0
	for image_path in image_list:
		images[i] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).reshape(height, width, num_channels)
		i += 1
	
	return images

def train_test_split(X, y, rate = 0.7):
	split_point = int(X.shape[0] * rate)
	X_train = X[ : split_point]
	X_validation = X[split_point : ]
	y_train = y[ : split_point]
	y_validation = y[split_point : ]
	return X_train , X_validation, y_train, y_validation	
