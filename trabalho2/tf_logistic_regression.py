#from __future__ import absolute_import
import numpy as np
import random
import sys
import time
import copy
import os
import tensorflow as tf

import dataset_manip	

def main():
	DATASET_DIRECTORY = '/home/felipe/tcv3/data_part1'
	TRAIN_RATE = 0.8
	NUM_EPOCHS = 1000
	BATCH_SIZE = 500
	MODEL_PATH = ''
	
	X, y, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
	num_classes = len(set(y))
	
	X_train, X_validation, y_train, y_validation = dataset_manip.split_dataset(X, y, rate = TRAIN_RATE)
	
	graph = tf.Graph()
	with graph.as_default():
		global_step = tf.Variable(0, trainable = False)
			
		X_tf = tf.placeholder(tf.float32, shape = (None, X.shape[1]))
		y_tf = tf.placeholder(tf.int64, shape = (None,))
		y_one_hot = tf.one_hot(y_tf, num_classes)
		learning_rate = tf.train.exponential_decay(1e-3, global_step, 250, 0.93, staircase = True)

		out = tf.layers.dense(X_tf, num_classes, activation = tf.nn.softmax)
		loss = tf.reduce_mean(tf.reduce_sum((y_one_hot - out) ** 2))

		train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
		result = tf.argmax(out, 1)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(result, y_tf), tf.float32))
		
		saver = tf.train.Saver()
	
	def train(X_train, y_train):
		best_acc = 0
		
		with tf.Session(graph = graph) as session:
			session.run(tf.global_variables_initializer())
			
			index = 0
			X_remainder = np.empty(shape = (0, X_train.shape[1]))
			y_remainder = np.empty(shape = (0, ))
			num_samples = X_train.shape[0]
			
			for epoch in range(1, NUM_EPOCHS + 1):
				print('Epoch {}'.format(epoch))
				
				while index < num_samples:
					session.run([train_op], feed_dict = {X_tf: X_train[index : index + BATCH_SIZE], y_tf: y_train[index : index + BATCH_SIZE]})
					
					if index + BATCH_SIZE > num_samples:
						X_remainder = np.copy(X_train[index : ])
						y_remainder = np.copy(y_train[index : ])
						
					index += BATCH_SIZE
				
				index = (index % num_samples)
				
				p = np.random.permutation(num_samples)
				X_train = X_train[p]
				y_train = y_train[p]
				
				if (X_remainder.shape[0] > 0):
					X_remainder = np.concatenate((X_remainder, X_train[ : index]), axis = 0)
					y_remainder = np.concatenate((y_remainder, y_train[ : index]), axis = 0)	
					session.run([train_op], feed_dict = {X_tf: X_remainder, y_tf: y_remainder})
					
					assert X_remainder.shape[0] == BATCH_SIZE
					assert y_remainder.shape[0] == BATCH_SIZE
				
					
				print(session.run(learning_rate))	
				val_loss, val_accuracy = session.run([loss, accuracy], feed_dict = {X_tf: X_validation, y_tf: y_validation})
				if (val_accuracy > best_acc):
					#saver.save(session, './my-model')
					best_acc = val_accuracy
									
				print('Validation Loss: {:.3f}\tValidation Accuracy: {:.3f}\t Best Acc: {:.3f}'.format(val_loss, val_accuracy, best_acc))
				
			
			print(best_acc)	
	
	train(X_train, y_train)
	
	#with tf.Session(graph = graph) as sess:
	#	saver.restore(sess, './my-model')
	#	val_loss, val_accuracy = sess.run([loss, accuracy], feed_dict = {X_tf: X_validation, y_tf: y_validation})
	#	print('Validation Loss: {:.3f}\tValidation Accuracy: {:.3f}'.format(val_loss, val_accuracy))

	
if __name__ == '__main__':
	main()
