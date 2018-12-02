import sys
import os
import numpy as np

import dataset_manip
from model import Model
from ensemble import Ensemble

def load_directory(directory_path):
	filenames = dataset_manip.get_filenames(directory_path)
	
	for i, filename in enumerate(filenames):
		filenames[i] = os.path.join(directory_path, filename)
		
	return filenames

def main():
	#print(sys.argv)
	test_set_path = sys.argv[1]
	output_file_path = sys.argv[2]

	X_test = dataset_manip.load_images(load_directory(test_set_path))/255
	
	#model = Model(image_shape = (77, 71, 1), num_classes = 10, model_path = './model_files/model', batch_size = 512, first_run = False)
	#dataset_manip.store_predictions(dataset_manip.get_filenames(test_set_path), model.predict(X_test), output_file_path)

	ens = Ensemble(input_shape = (77, 71, 1), num_classes = 10, num_models = 11, batch_size = 512, path = './ensemble_files', load = True)
	dataset_manip.store_predictions(dataset_manip.get_filenames(test_set_path), ens.predict(X_test), output_file_path)	
	
if __name__ == '__main__':
	main()
