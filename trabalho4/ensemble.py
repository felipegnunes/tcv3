import numpy as np
import collections
import dataset_manip
from model import Model

class Ensemble:

	def __init__(self, input_shape, num_classes, num_models, batch_size, path = '.', load = False):
		self.num_models = num_models
		self.batch_size = batch_size
		
		self.models = []
		for i in range(num_models):
			model = Model(image_shape = input_shape, 
				      num_classes = num_classes, 
				      model_path = path + '/model_' + str(i), 
				      batch_size = batch_size, 
				      first_run = not load)
			self.models.append(model)
		
	def train(self, X, y, epochs_per_model, split_rate):
		num_samples = X.shape[0]
		
		for i in range(self.num_models):
			print('Training model {}'.format(i))
			permutation = np.random.permutation(num_samples)
			X = X[permutation]
			y = y[permutation]
			X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = split_rate)
			self.models[i].train(X_train, y_train, X_validation, y_validation, epochs_per_model)
			
	def predict(self, X):
		votes = np.empty(shape = (X.shape[0], self.num_models))
		votes[:, 0] = self.models[0].predict(X)
		
		for i in range(1, self.num_models):
			votes[:, i] = self.models[i].predict(X)
		
		predictions = np.empty(shape = (X.shape[0], ), dtype = np.int32)
		for i in range(X.shape[0]):
			predictions[i] = collections.Counter(votes[i, :]).most_common(1)[0][0]	
			
		return predictions	
		
		
	def measure_accuracy(self, X, y):
		return np.mean(self.predict(X) == y)
