import tensorflow as tf
import numpy as np

import dataset_manip
import image_manip
import dataset

class Model:
	
	def __init__(self, image_shape, num_classes, model_path, batch_size = 8, first_run = True):
		self.img_height, self.img_width, self.img_num_channels = image_shape
		self.model_path = model_path
		self.batch_size = batch_size
		self.num_classes = num_classes
		
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.X = tf.placeholder(tf.float32, shape = (None, self.img_height, self.img_width, self.img_num_channels), name = 'X')
			self.y = tf.placeholder(tf.int32, shape = (None, ), name = 'y')
			self.y_one_hot = tf.one_hot(self.y, num_classes)
			self.keep_prob0 = tf.placeholder_with_default(1.0, shape = ())
			self.keep_prob1 = tf.placeholder_with_default(1.0, shape = ())
			
			self.output = tf.layers.conv2d(self.X, filters = 6, kernel_size = (5, 5), strides = (1, 1), padding = 'same', activation = tf.nn.relu)
			self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'same')
			self.output = tf.nn.dropout(self.output, self.keep_prob0)			
			
			self.output = tf.layers.conv2d(self.output, filters = 16, kernel_size = (5, 5), strides = (1, 1), padding = 'same', activation = tf.nn.relu)
			self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'same')
			self.output = tf.nn.dropout(self.output, self.keep_prob0)
			
			self.output = tf.layers.conv2d(self.output, filters = 120, kernel_size = (5, 5), strides = (1, 1), padding = 'same', activation = tf.nn.relu)
			self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'same')
			self.output = tf.nn.dropout(self.output, self.keep_prob0)
			
			num_elements = (self.output.shape[1] * self.output.shape[2] * self.output.shape[3])
			self.output = tf.reshape(self.output, (-1, num_elements))
			self.output = tf.layers.dense(self.output, 1024, activation = tf.nn.relu)
			self.output = tf.nn.dropout(self.output, self.keep_prob1)
			self.output = tf.layers.dense(self.output, 128, activation = tf.nn.relu)
			self.output = tf.nn.dropout(self.output, self.keep_prob1)
			self.output = tf.layers.dense(self.output, num_classes, activation = tf.nn.softmax)
			
			self.result = tf.argmax(self.output, 1, output_type = tf.int32)
			self.loss = tf.losses.softmax_cross_entropy(self.y_one_hot, self.output)
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.result, self.y), tf.float32), name = 'accuracy')
			
			self.train_operation = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(self.loss)
			self.saver = tf.train.Saver()
			
			if first_run:
				with tf.Session(graph = self.graph) as session:
					session.run(tf.global_variables_initializer())
					self.saver.save(session, self.model_path)
			
	def train(self, X_train, y_train, X_validation, y_validation, num_epochs):
		best_accuracy = self.measure_accuracy(X_validation, y_validation)
		train_set = dataset.Dataset(X_train, y_train, self.batch_size)
		
		with tf.Session(graph = self.graph) as session:
			self.saver.restore(session, self.model_path)
			
			for epoch in range(1, num_epochs + 1):
				print('Epoch {}'.format(epoch))
				training_loss = 0
				training_accuracy = 0
				num_batches = 0
				
				on_last_batch = False
				while not on_last_batch:
					on_last_batch = train_set.on_last_batch()
					X_batch, y_batch = train_set.next_batch(augment = True)
					batch_loss, batch_accuracy, _ = session.run([self.loss, self.accuracy, self.train_operation],
																feed_dict = {self.X: X_batch, self.y: y_batch, self.keep_prob0: 0.75, self.keep_prob1: 0.5})
					training_loss += batch_loss
					training_accuracy += batch_accuracy
					num_batches += 1
					
				training_loss /= num_batches
				training_accuracy /= num_batches
				validation_accuracy = self._measure_accuracy_online(session, X_validation, y_validation)
	
				print('Training Accuracy:   {:8.5}\tTraining Loss: {}'.format(training_accuracy, training_loss))
				print('Validation Accuracy: {:8.5}{}'.format(validation_accuracy, '\t(ACCURACY IMPROVED)' if validation_accuracy > best_accuracy else ''))
				
				if (validation_accuracy > best_accuracy):
					best_accuracy = validation_accuracy
					self.saver.save(session, self.model_path)
	
	def train_unsupervised(self, X_train, X_validation, y_validation, num_epochs):
		best_accuracy = self.measure_accuracy(X_validation, y_validation)
		
		with tf.Session(graph = self.graph) as session:
			self.saver.restore(session, self.model_path)
			
			for epoch in range(1, num_epochs + 1):
				print('Epoch {}'.format(epoch))
				training_loss = 0
				training_accuracy = 0
				num_batches = 0
				
				train_set = dataset.Dataset(X_train, self._predict_online(session, X_train), self.batch_size)
				
				on_last_batch = False
				while not on_last_batch:
					on_last_batch = train_set.on_last_batch()
					X_batch, y_batch = train_set.next_batch()
					batch_loss, batch_accuracy, _ = session.run([self.loss, self.accuracy, self.train_operation],
																feed_dict = {self.X: X_batch, self.y: y_batch, self.keep_prob0: 0.75, self.keep_prob1: 0.5})
					training_loss += batch_loss
					training_accuracy += batch_accuracy
					num_batches += 1
					
				training_loss /= num_batches
				training_accuracy /= num_batches
				validation_accuracy = self._measure_accuracy_online(session, X_validation, y_validation)
	
				print('Training Accuracy:   {:8.5}\tTraining Loss: {}'.format(training_accuracy, training_loss))
				print('Validation Accuracy: {:8.5}{}'.format(validation_accuracy, '\t(ACCURACY IMPROVED)' if validation_accuracy > best_accuracy else ''))
				
				if (validation_accuracy > best_accuracy):
					best_accuracy = validation_accuracy
					self.saver.save(session, self.model_path)
	
	def _predict_online(self, session, X):
		num_samples = X.shape[0]
			
		y = np.empty(shape = (num_samples, ), dtype = np.int32)
		for i in range(0, num_samples, self.batch_size):
				j = min(i + self.batch_size, num_samples)
				y[i : j] = session.run(self.result, feed_dict = {self.X: X[i : j]})
	
		return y
			
	def _measure_accuracy_online(self, session, X_validation, y_validation):
		num_validation_samples = X_validation.shape[0]
		predictions = np.empty(shape = (num_validation_samples, ), dtype = np.int32)
		for i in range(0, num_validation_samples, self.batch_size):
			j = min(i + self.batch_size, num_validation_samples)
			predictions[i : j] = session.run(self.result, feed_dict = {self.X: X_validation[i : j]})
		return np.mean(predictions == y_validation)
	
	def predict(self, X):
		num_samples = X.shape[0]
		predictions = np.empty(shape = (num_samples, ), dtype = np.int32)
		
		with tf.Session(graph = self.graph) as session:
			self.saver.restore(session, self.model_path)
			
			for i in range(0, num_samples, self.batch_size):
				j = min(i + self.batch_size, num_samples)
				predictions[i : j] = session.run(self.result, feed_dict = {self.X: X[i : j]})
		
		return predictions
	
	def measure_accuracy(self, X, y):
		return np.mean(self.predict(X) == y)
	
