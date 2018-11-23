import os
import cv2
import numpy as np
import random

def load_dataset(dataset_directory, shuffle = True):
	labeled_set, hidden_set = load_paths(dataset_directory)
		
	if shuffle:
		random.shuffle(labeled_set)
	
	X_hidden = load_images(hidden_set)
	X = load_images([row[0] for row in labeled_set])
	
	y = np.array([row[1] for row in labeled_set], dtype = np.int32)
	
	X_hidden /= 255
	X /= 255
	
	return X, y, X_hidden	
	
def load_paths(dataset_directory):
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
	
	images = np.empty([num_images, height, width, num_channels], dtype = np.float32)
	
	i = 0
	for image_path in image_list:
		images[i] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).reshape(height, width, num_channels)
		i += 1
	
	return images

def store_predictions(filenames, predictions, path = './predictions.txt'):
	result = list(zip(filenames, predictions))
	result.sort(key = lambda x: int(os.path.splitext(x[0])[0]))
	
	with open(path, 'w') as f:
		for filename, prediction in result:
			f.write('{} {}\n'.format(filename, prediction))

def split_dataset(X, y, rate = 0.7):
	split_point = int(X.shape[0] * rate)
	X1 = X[ : split_point]
	X2 = X[split_point : ]
	y1 = y[ : split_point]
	y2 = y[split_point : ]
	return X1, X2, y1, y2	

def get_immediate_subdirectories(directory_path):
	return [subdirectory for subdirectory in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, subdirectory))]

def get_filenames(directory_path):
	return [filename for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]

