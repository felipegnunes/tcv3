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

def define_model(params):
	graph = tf.Graph()
	
	with graph.as_default():
		global_step = tf.Variable(0, trainable = False)
		
		prob = tf.placeholder_with_default(1.0, shape = (), name = 'prob')
		
		X_tf = tf.placeholder(tf.float32, shape = (None, params['num_inputs']), name = 'X')
		y_tf = tf.placeholder(tf.int64, shape = (None,), name = 'y')
		y_one_hot = tf.one_hot(y_tf, params['num_outputs'])
		learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step, params['decay_steps'], params['decay_rate'], staircase = params['staircase'], name = 'learning_rate')

		out = tf.layers.dense(tf.nn.dropout(X_tf, prob), params['num_outputs'], activation = params['activation_function'], name = 'output')
		result = tf.argmax(out, 1, name = 'predict')
		loss = tf.reduce_mean(tf.reduce_sum((y_one_hot - out) ** 2), name = 'loss')
		
		train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate, name = 'train_op').minimize(loss, global_step = global_step)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(result, y_tf), tf.float32), name = 'accuracy')

		saver = tf.train.Saver(name = 'saver')	

	return graph

def train_model(graph, X_train, X_validation, y_train, y_validation, num_epochs, batch_size, dropout_keep_prob, X_hidden):
	best_acc = 0
	
	with tf.Session(graph = graph) as session:
		session.run(tf.global_variables_initializer())
		
		index = 0
		X_remainder = np.empty(shape = (0, X_train.shape[1]))
		y_remainder = np.empty(shape = (0, ))
		num_samples = X_train.shape[0]
		
		for epoch in range(1, num_epochs + 1):
			print('Epoch {}'.format(epoch))
			
			while index < num_samples:
				session.run([graph.get_operation_by_name('train_op')], 
					     feed_dict = {graph.get_tensor_by_name('X:0'): X_train[index : index + batch_size],
							  graph.get_tensor_by_name('y:0'): y_train[index : index + batch_size],
							  graph.get_tensor_by_name('prob:0'): dropout_keep_prob})
				
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
				
				session.run([graph.get_operation_by_name('train_op')], 
					     feed_dict = {graph.get_tensor_by_name('X:0'): X_remainder,
							  graph.get_tensor_by_name('y:0'): y_remainder,
							  graph.get_tensor_by_name('prob:0'): dropout_keep_prob})
				
				assert X_remainder.shape[0] == batch_size
				assert y_remainder.shape[0] == batch_size
			
				
			val_loss, val_acc = session.run([graph.get_tensor_by_name('loss:0'), graph.get_tensor_by_name('accuracy:0')],
							 feed_dict = {graph.get_tensor_by_name('X:0'): X_validation,
							 	      graph.get_tensor_by_name('y:0'): y_validation})
			
			if (val_acc > best_acc):
				best_acc = val_acc
				predictions = session.run([graph.get_tensor_by_name('predict:0')],
							 feed_dict = {graph.get_tensor_by_name('X:0'): X_hidden})
							 								
			print('Val. Loss: {:.3f}\tVal. Accuracy: {:.3f}\tBest Accuracy: {:.3f}'.format(val_loss, val_acc, best_acc))
		
	return best_acc, predictions
	
def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	DATASET_DIRECTORY = '/home/felipe/tcv3/data_part1'
	TRAIN_RATE = 0.8
	NUM_EPOCHS = 2000
	BATCH_SIZE = 500
	MODEL_PATH = ''
	
	X, y, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
	num_classes = len(set(y))
	
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = TRAIN_RATE)
	
	graph = define_model({'num_inputs': X.shape[1],
			      'num_outputs': num_classes,
			      'learning_rate': 1e-3,
			      'decay_steps': 300,
			      'decay_rate': 0.95,
			      'staircase': True,
			      'activation_function': tf.nn.softmax})
	
	best_acc, predictions = train_model(graph, X_train, X_validation, y_train, y_validation, NUM_EPOCHS, BATCH_SIZE, 0.5, X_hidden)
	
	_, test_images = dataset_manip.load_paths(DATASET_DIRECTORY)
	
	result = list(zip(test_images, predictions[0]))
	
	result.sort(key = lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
	print('Number of predictions: {}'.format(len(result)))
	
	with open('logistic_regression.txt', 'w') as f:
		for image_name, prediction in result:
			f.write('{} {}\n'.format(os.path.basename(image_name), prediction))
	
	#decay_rates = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
	#decay_steps = [250, 300, 350, 400]
	#dropouts = [0.5, 0.7, 0.9, 1.0]
	#configurations = list(itertools.product(decay_rates, decay_steps, dropouts))
	#num_configurations = len(configurations)
	
	#best_acc = 0
	#i = 0
	
	#for decay_rate, decay_step, dropout in configurations:
	#	i += 1
	#	print('Configuration {} of {}'.format(i, num_configurations))
	#	graph = define_model({'num_inputs': X.shape[1],
	#		      'num_outputs': num_classes,
	#		      'learning_rate': 1e-3,
	#		      'decay_steps': decay_step,
	#		      'decay_rate': decay_rate,
	#		      'staircase': True,
	#		      'activation_function': tf.nn.softmax})
		
	#	acc = train_model(graph, X_train, X_validation, y_train, y_validation, NUM_EPOCHS, BATCH_SIZE, dropout)
	#	if (acc > best_acc):
	#		best_acc = acc
	#		best_configuration = decay_rate, decay_step, dropout
		
	#	print('Best Found: ' + str(best_acc))
		
	#print('Best acc: {}\nBest config.: {}'.format(best_acc, best_configuration))
		
	#with tf.Session(graph = graph) as sess:
	#	saver.restore(sess, './my-model')
	#	val_loss, val_accuracy = sess.run([loss, accuracy], feed_dict = {X_tf: X_validation, y_tf: y_validation})
	#	print('Validation Loss: {:.3f}\tValidation Accuracy: {:.3f}'.format(val_loss, val_accuracy))

	
if __name__ == '__main__':
	main()
