import os
import cv2
import numpy as np
import random
import dataset_manip	
import math

def shift(image, horizontal_shift, vertical_shift):
	height, width = image.shape[ : 2]
	M = np.float32([[1, 0, horizontal_shift], 
			[0, 1, vertical_shift]])
	return cv2.warpAffine(image, M, (width, height)).reshape(image.shape)
	
def rotate(image, angle):
	height, width = image.shape[ : 2]
	M = cv2.getRotationMatrix2D(center = (width/2, height/2), angle = angle, scale = 1)
	return cv2.warpAffine(image, M, (width, height)).reshape(image.shape)

def increase_contrast(image, alpha):
	return np.minimum(image * alpha, 255)
	
def zoom(image, zoom_factor):
	height, width = image.shape[ : 2]
	
	if (int(zoom_factor * height) == height and int(zoom_factor * width) == width):
		return image
	
	new_image = np.zeros(shape = (height, width))
	if (zoom_factor < 1.0):
		zoomed_image = cv2.resize(image, (int(zoom_factor * width), int(zoom_factor * height)), interpolation = cv2.INTER_AREA)
		zoomed_height, zoomed_width = zoomed_image.shape[ : 2]
		new_image[height//2 + 1 - math.ceil(zoomed_height/2) : height//2 + math.floor(zoomed_height/2) + 1, 
			  width//2 + 1 - math.ceil(zoomed_width/2) : width//2 + math.floor(zoomed_width/2) + 1] = zoomed_image 
	else:	 
		zoomed_image = cv2.resize(image, (int(zoom_factor * width), int(zoom_factor * height)), interpolation = cv2.INTER_CUBIC)
		zoomed_height, zoomed_width = zoomed_image.shape[ : 2]
		new_image = zoomed_image[zoomed_height//2 + 1 - math.ceil(height/2) : zoomed_height//2 + math.floor(height/2) + 1, 
			 		 zoomed_width//2 + 1 - math.ceil(width/2) : zoomed_width//2 + math.floor(width/2) + 1]
	
	return new_image.reshape(image.shape)
			 
def perturbate_randomly(images, horizontal_shift_range, vertical_shift_range, angle_range, contrast_alpha_range, zoom_factor_range):
	num_images = images.shape[0]
	new_images = np.copy(images)
	
	for i in range(num_images):
		if random.random() >= 0.5:
			horizontal_shift = random.uniform(horizontal_shift_range[0], horizontal_shift_range[1])
			vertical_shift = random.uniform(vertical_shift_range[0], vertical_shift_range[1])
			new_images[i] = shift(new_images[i], horizontal_shift, vertical_shift)
		if random.random() >= 0.5:
			angle = random.uniform(angle_range[0], angle_range[1])
			new_images[i] = rotate(new_images[i], angle)
		if random.random() >= 0.5:
			alpha = random.uniform(contrast_alpha_range[0], contrast_alpha_range[1])
			new_images[i] = increase_contrast(new_images[i], alpha)
		if random.random() >= 0.5:
			zoom_factor = random.uniform(zoom_factor_range[0], zoom_factor_range[1])
			new_images[i] = zoom(new_images[i], zoom_factor)
	
	return new_images
	
def save_image_set(images, path):
	num_images = images.shape[0]
	images = perturbate_randomly(images, (-10, 10), (-10, 10), (-15, 15), (.9, 1.1), (.9, 1.2))
	for i in range(num_images):
		cv2.imwrite(os.path.join(path, str(i) + '.png'), to_byte_format(images[i]))

def to_byte_format(image):
	return np.minimum(image * 255, 255)
	
def main():
	X, y, X_hidden = dataset_manip.load_dataset('/home/felipe/tcv3/data_part1', False)
	
	save_image_set(X, '/home/felipe/tcv3/data_test')
	
if __name__ == '__main__':
	main()
