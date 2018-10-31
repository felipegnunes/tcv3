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

class Model:
	
	def __init__(self, image_shape, num_classes):
		self.img_height, self.img_width, self.img_num_channels = image_shape
		
		self.graph = tf.Graph()
		
		with self.graph.as_default():
			self.X = tf.placeholder(tf.float32, shape = (None, self.img_height, self.img_width, self.img_num_channels), name = 'X')
			self.y = tf.placeholder(tf.int32, shape = (None, ), name = 'y')
			self.y_one_hot = tf.one_hot(self.y, num_classes)
			self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
			
			print(self.X.shape)
			self.output = tf.layers.conv2d(self.X, filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', activation = tf.nn.relu)
			print(self.output.shape)
			self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'valid')
			print(self.output.shape)
			self.output = tf.layers.conv2d(self.X, filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', activation = tf.nn.relu)
			print(self.output.shape)
			self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'valid')
			print(self.output.shape)
			
			num_elements = self.output.shape[1] * self.output.shape[2] * self.output.shape[3]
			self.output = tf.reshape(self.output, (-1, num_elements))
			self.output = tf.layers.dense(self.output, num_classes, activation = tf.nn.softmax)
			print(self.output.shape)
			
			self.result = tf.argmax(self.output, 1, output_type = tf.int32)
			print(self.result.shape)
			self.loss = tf.losses.softmax_cross_entropy(self.y_one_hot, self.output)
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.result, self.y), tf.float32), name = 'accuracy')
			
			self.train_operation = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
			self.saver = tf.train.Saver()
			
	def train(self, X_train, y_train, X_validation, y_validation, num_epochs, batch_size, X_hidden):
		self.session = tf.Session(graph = self.graph)
		best_acc = 0
		
		with self.session as session:
			session.run(tf.global_variables_initializer())
		
			index = 0
			X_remainder = np.empty(shape = (0, self.img_height, self.img_width, self.img_num_channels))
			y_remainder = np.empty(shape = (0, ))
			num_samples = X_train.shape[0]
			
			for epoch in range(1, num_epochs + 1):
				print('Epoch {}'.format(epoch))
				
				while index < num_samples:
					session.run([self.train_operation], feed_dict = {self.X: X_train[index : index + batch_size], 
											 self.y: y_train[index : index + batch_size], 
											 self.learning_rate: 1e-3})#self.keep_prob: 0.5})
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
											 self.learning_rate: 1e-3})#self.keep_prob: 0.5})
					assert X_remainder.shape[0] == batch_size
					assert y_remainder.shape[0] == batch_size
				
				learning_rate, validation_loss, validation_accuracy = session.run([self.learning_rate, self.loss, self.accuracy], feed_dict = {self.X: X_validation, 
																			       self.y: y_validation,
																			       self.learning_rate: 1e-3})
				print('Learning Rate: {}\tValidation Loss: {}\tValidation Accuracy: {}'.format(learning_rate, validation_loss, validation_accuracy))
				if (validation_accuracy > best_acc):
					best_acc = validation_accuracy
					predictions = session.run([self.result], feed_dict = {self.X: X_hidden[:1000]})
					self.save('./test')
					
				
		return best_acc, predictions
	
	#def predict(self, X, batch_size):
	#	index = 0
	#	X_remainder = np.empty(shape = (0, self.img_height, self.img_width, self.img_num_channels))
	#	predictions = np.empty(shape = (0, ))
	#	num_samples = X.shape[0]
	#			
	#	for i in range(0, num_samples, batch_size):
	#		predictions = np.concatenate(predictions, self.session.run([self.result],
	#									   feed_dict = {self.X: X[i : min(i + batch_size, num_samples)]}
	#									  ))
	#	return predictions
		
	def load(self):
		pass
	
	def save(self, path):
		self.saver.save(self.session, path)
		print('Saving to ' + path)
		
def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	DATASET_DIRECTORY = '/home/felipe/tcv3/data_part1'
	TRAIN_RATE = 0.8
	NUM_EPOCHS = 100
	BATCH_SIZE = 20
	
	X, y, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
	num_classes = len(set(y))
	
	print(X.shape)
	print(X_hidden.shape)
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = TRAIN_RATE)

	model = Model(X.shape[1 : ], num_classes)
	model.train(X_train, y_train, X_validation, y_validation, NUM_EPOCHS, BATCH_SIZE, X_hidden)
	
	print(model.predict(X_hidden, 10))
	
if __name__ == '__main__':
	main()
