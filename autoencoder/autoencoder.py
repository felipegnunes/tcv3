import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Autoencoder:
	
	def __init__(self, input_shape, model_path, batch_size, first_run = True):
		self.image_height, self.image_width, self.image_num_channels = input_shape
		self.model_path = model_path
		self.batch_size = batch_size
		
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.X = tf.placeholder(tf.float32, shape = (None, self.image_height, self.image_width, self.image_num_channels), name = 'X')
			
			print('Input shape: ' + str(self.X.shape))
			self.output = tf.layers.conv2d(self.X, filters = 8, kernel_size = (5, 5), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
			print(self.output.shape)
			
			self.output = tf.layers.conv2d(self.output, filters = 16, kernel_size = (5, 5), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
			print('Inter. shape: ' + str(self.output.shape))
			
			self.output = tf.layers.conv2d_transpose(self.output, filters = 8, kernel_size = (5, 5), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
			print(self.output.shape)
			
			self.output = tf.layers.conv2d_transpose(self.output, filters = 1, kernel_size = (5, 5), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
			print('Output shape: ' + str(self.output.shape))
			
			
			self.loss = tf.reduce_sum((self.output - self.X) ** 2)
			
			self.train_operation = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(self.loss)
			self.saver = tf.train.Saver()

			if first_run:
				with tf.Session(graph = self.graph) as session:
					session.run(tf.global_variables_initializer())
					self.saver.save(session, self.model_path)
	
	def measure_loss(self, X):
		num_samples = X.shape[0]
		loss = 0
		batches = 0
		with tf.Session(graph = self.graph) as session:
			self.saver.restore(session, self.model_path)
			for i in range(0, num_samples, self.batch_size):
				j = min(i + self.batch_size, num_samples)
				loss += session.run(self.loss, feed_dict = {self.X: X[i : j]})
				batches += 1
		return loss/batches
			
	def train(self, X_train, X_validation, num_epochs):
		best_loss = self.measure_loss(X_validation)
		
		with tf.Session(graph = self.graph) as session:
			self.saver.restore(session, self.model_path)
			
			index = 0
			num_samples = X_train.shape[0]
			X_remainder = np.empty(shape = (0, self.image_height, self.image_width, self.image_num_channels))
			
			for epoch in range(1, num_epochs + 1):
				training_loss = 0
				num_batches = 0
				
				print('Epoch {}'.format(epoch))
				
				while index < num_samples:
					batch_loss, _ = session.run([self.loss, self.train_operation], feed_dict = {self.X: X_train[index : index + self.batch_size]})
					training_loss += batch_loss
					num_batches += 1
										       		
					if index + self.batch_size > num_samples:
						X_remainder = np.copy(X_train[index : ])
					
					index += self.batch_size
				
				index = (index % num_samples)
				
				permutation = np.random.permutation(num_samples)
				X_train = X_train[permutation]
				
				if (X_remainder.shape[0] > 0):
					X_remainder = np.concatenate((X_remainder, X_train[ : index]), axis = 0)
					batch_loss, _ = session.run([self.loss, self.train_operation], feed_dict = {self.X: X_remainder})
					training_loss += batch_loss
					num_batches += 1
				
				training_loss /= num_batches
				
				num_validation_samples = X_validation.shape[0]
				validation_loss = 0
				validation_batches = 0
				for i in range(0, num_validation_samples, self.batch_size):
					j = min(i + self.batch_size, num_validation_samples)
					validation_loss += session.run(self.loss, feed_dict = {self.X: X_validation[i : j]})
					validation_batches += 1
				validation_loss /= validation_batches
					
				print('Training Loss:   {:8.5}'.format(training_loss))
				print('Validation Loss: {:8.5}'.format(validation_loss))
				
				if (validation_loss < best_loss):
					best_loss = validation_loss
					print('New best loss:   {:8.5}'.format(best_loss))
					self.saver.save(session, self.model_path)
				
				model_result = session.run(self.output, feed_dict = {self.X: X_validation[ : 1]})[0]
				print(model_result.shape)
				result = np.concatenate((model_result, X_validation[0]), axis = 1)	
				plt.imshow(result.reshape(64, 128), cmap = "gray")
				plt.show()
		
	def predict(self):
		pass
