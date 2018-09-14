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

def get_file_number(directory_path):
	return len(get_filenames(directory_path))

def load_filenames(dataset_directory):
	train_dir = os.path.join(dataset_directory, 'train')
	test_dir

def load_images():
	pass
	
def main():
	# PARAMETERS
	train_dir = sys.argv[1] #'/tmp/guest-jbsthn/topicos_compvis3/data_part1/train'
	test_dir = sys.argv[2] #'/tmp/guest-jbsthn/topicos_compvis3/data_part1/test'
	train_test_rate = float(sys.argv[3]) #0.7
	
	# LOADING IMAGE NAMES AND LABELS
	
	directory_list = get_immediate_subdirectories(train_dir)

	imgs = []

	for directory in directory_list:
		directory_path = os.path.join(train_dir, directory)
		label = int(directory)
		image_list = get_filenames(directory_path)
		
		for image in image_list:
			image_path = os.path.join(directory_path, image)
			imgs.append((image_path, label))
	
	# SHUFFLING IMAGES
		
	random.shuffle(imgs)
	
	# LOADING IMAGES ON A NUMPY ARRAY
	
	sample_img_path = imgs[0][0]
	example_image = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
	
	if (len(example_image.shape) < 3):
		height, width = example_image.shape
		num_channels = 1
	else:
		height, width, num_channels = example_image.shape
			
	split_point = int(train_test_rate * len(imgs))
	num_train_imgs = split_point
	num_test_imgs = len(imgs) - split_point
	
	X_train = np.empty([num_train_imgs, height, width, num_channels], dtype=np.uint8)
	X_test = np.empty([num_test_imgs, height, width, num_channels], dtype=np.uint8)
	y_train = np.empty([num_train_imgs], dtype=np.int64)
	y_test = np.empty([num_test_imgs], dtype=np.int64)
	
	i = 0
	for image_path, label in imgs[ : split_point]:	
		X_train[i] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).reshape(height, width, num_channels)
		y_train[i] = label
		i += 1
	
	i = 0
	for image_path, label in imgs[split_point : ]:
		X_test[i] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).reshape(height, width, num_channels)
		y_test[i] = label
		i += 1
		
	print('X_train: {}\nX_test: {}\ny_train: {}\ny_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))	
		
	cv2.imshow('x_train', X_train[0])
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
	
if __name__ == '__main__':
	main()
