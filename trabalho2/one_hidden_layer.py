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

class OneHiddenLayer:
	
	def __init__(self, num_inputs, num_outputs, hidden_layer_size, learning_rate, decay_steps, decay_rate):
		self.graph = tf.Graph()
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.hidden_layer_size = hidden_layer_size
		
		with self.graph.as_default():
			self.X = tf.placeholder(tf.float32, shape = (None, num_inputs), name = 'X')
			self.y = tf.placeholder(tf.int64, shape = (None, ), name = 'y')
			self.y_one_hot = tf.one_hot(self.y, num_outputs)
			
			self.global_step = tf.Variable(0, trainable = False)
			self.keep_prob = tf.placeholder_with_default(1.0, shape = ())
			self.learning_rate = tf.train.exponential_decay(learning_rate = learning_rate, 
									global_step = self.global_step, 
									decay_steps = decay_steps, 
									decay_rate = decay_rate, 
									staircase = True, 
									name = 'learning_rate'
									)
			
			self.hidden_layer = tf.layers.dense(tf.nn.dropout(self.X, self.keep_prob), hidden_layer_size, activation = tf.nn.relu, name = 'hidden_layer')
			self.output_layer = tf.layers.dense(self.hidden_layer, num_outputs, activation = tf.nn.softmax, name = 'output_layer')
			
			self.result = tf.argmax(self.output_layer, 1)
			self.loss = tf.losses.softmax_cross_entropy(self.y_one_hot, self.output_layer) #tf.losses.mean_squared_error(labels = self.y_one_hot, predictions = self.output_layer)
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.result, self.y), tf.float32), name = 'accuracy')
			
			self.train_operation = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate, name = 'train_operation').minimize(self.loss, global_step = self.global_step) 
		
	
	def train(self, X_train, y_train, X_validation, y_validation, batch_size, num_epochs, X_hidden):
		self.session = tf.Session(graph = self.graph)
		best_acc = 0
		
		with self.session as session:
			session.run(tf.global_variables_initializer())
		
			index = 0
			X_remainder = np.empty(shape = (0, self.num_inputs))
			y_remainder = np.empty(shape = (0, ))
			num_samples = X_train.shape[0]
			
			for epoch in range(1, num_epochs + 1):
				print('Epoch {}'.format(epoch))
				
				while index < num_samples:
					session.run([self.train_operation], feed_dict = {self.X: X_train[index : index + batch_size], 
											 self.y: y_train[index : index + batch_size], 
											 self.keep_prob: 0.5})
					if index + batch_size > num_samples:
						X_remainder = np.copy(X_train[index : ])
						y_remainder = np.copy(y_train[index : ])
					
					index += batch_size
				
				index = (index % num_samples)
				
				p = np.random.permutation(num_samples)
				X_train = X_train[p]
				y_train = y_train[p]
				
				if (X_remainder.shape[0] > 0):
					X_remainder = np.concatenate((X_remainder, X_train[ : index]), axis = 0)
					y_remainder = np.concatenate((y_remainder, y_train[ : index]), axis = 0)	
					session.run([self.train_operation], feed_dict = {self.X: X_remainder, 
											 self.y: y_remainder, 
											 self.keep_prob: 0.5})
					assert X_remainder.shape[0] == batch_size
					assert y_remainder.shape[0] == batch_size
				
				learning_rate, validation_loss, validation_accuracy = session.run([self.learning_rate, self.loss, self.accuracy], feed_dict = {self.X: X_validation, 
																			       self.y: y_validation
																			       })
				print('Learning Rate: {}\tValidation Loss: {}\tValidation Accuracy: {}'.format(learning_rate, validation_loss, validation_accuracy))
				if (validation_accuracy > best_acc):
					best_acc = validation_accuracy
					predictions = session.run([self.result], feed_dict = {self.X: X_hidden})
				
		return best_acc, predictions	

def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	DATASET_DIRECTORY = '/home/felipe/tcv3/data_part1'
	TRAIN_RATE = 0.8
	NUM_EPOCHS = 2000
	BATCH_SIZE = 500
	HIDDEN_LAYER_SIZE = 400
	
	X, y, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
	num_classes = len(set(y))
	
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = TRAIN_RATE)

	#hl_sizes = []
	#learning_rates = []
	
	model = OneHiddenLayer(X.shape[1], num_classes, HIDDEN_LAYER_SIZE, 5e0, 200, 0.88)
	#model = OneHiddenLayer(X.shape[1], num_classes, HIDDEN_LAYER_SIZE, 4e0, 300, 0.85)
	#model.train(X_train, y_train, X_validation, y_validation, BATCH_SIZE, NUM_EPOCHS)
	
	best_acc, predictions = model.train(X_train, y_train, X_validation, y_validation, BATCH_SIZE, NUM_EPOCHS, X_hidden)
	print('Best: {}'.format(best_acc))
	
	_, test_images = dataset_manip.load_paths(DATASET_DIRECTORY)
	
	result = list(zip(test_images, predictions[0]))
	
	result.sort(key = lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
	print('Number of predictions: {}'.format(len(result)))
	
	with open('multilayer_perceptron.txt', 'w') as f:
		for image_name, prediction in result:
			f.write('{} {}\n'.format(os.path.basename(image_name), prediction))
	
if __name__ == '__main__':
	main()
