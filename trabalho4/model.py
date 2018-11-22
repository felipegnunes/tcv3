import tensorflow as tf
import numpy as np
import dataset_manip

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
			
			#self.output = tf.layers.conv2d(self.X, filters = 32, kernel_size = (7, 7), strides = (1, 1), padding = 'same', activation = tf.nn.relu)
			#self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'same')
			#self.output = tf.nn.dropout(self.output, self.keep_prob0)
			#print(self.output.shape)
			#self.output = tf.layers.conv2d(self.output, filters = 64, kernel_size = (5, 5), strides = (1, 1), padding = 'same', activation = tf.nn.relu)
			#self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'same')
			#self.output = tf.nn.dropout(self.output, self.keep_prob0)
			#print(self.output.shape)
			#self.output = tf.layers.conv2d(self.output, filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = tf.nn.relu)
			#self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'same')
			#self.output = tf.nn.dropout(self.output, self.keep_prob0)
			#print(self.output.shape)
			
			num_elements = (self.output.shape[1] * self.output.shape[2] * self.output.shape[3])
			self.output = tf.reshape(self.output, (-1, num_elements))
			self.output = tf.layers.dense(self.output, 120, activation = tf.nn.relu)
			self.output = tf.nn.dropout(self.output, self.keep_prob1)
			#self.output = tf.layers.dense(self.output, 64, activation = tf.nn.relu)
			#self.output = tf.nn.dropout(self.output, self.keep_prob1)
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
		
		with tf.Session(graph = self.graph) as session:
			self.saver.restore(session, self.model_path)
			
			index = 0
			X_remainder = np.empty(shape = (0, self.img_height, self.img_width, self.img_num_channels))
			y_remainder = np.empty(shape = (0, ))
			num_samples = X_train.shape[0]
			
			for epoch in range(1, num_epochs + 1):
				training_loss = 0
				training_accuracy = 0
				num_batches = 0
				
				print('Epoch {}'.format(epoch))
				
				while index < num_samples:
					batch_loss, batch_accuracy, _ = session.run([self.loss, self.accuracy, self.train_operation], 
										    feed_dict = {self.X: X_train[index : index + self.batch_size], 
												 self.y: y_train[index : index + self.batch_size],
												 self.keep_prob0: 0.75,
												 self.keep_prob1: 0.5})
					training_loss += batch_loss
					training_accuracy += batch_accuracy
					num_batches += 1
										       		
					if index + self.batch_size > num_samples:
						X_remainder = np.copy(X_train[index : ])
						y_remainder = np.copy(y_train[index : ])
					
					index += self.batch_size
				
				index = (index % num_samples)
				
				permutation = np.random.permutation(num_samples)
				X_train = X_train[permutation]
				y_train = y_train[permutation]
				
				if (X_remainder.shape[0] > 0):
					X_remainder = np.concatenate((X_remainder, X_train[ : index]), axis = 0)
					y_remainder = np.concatenate((y_remainder, y_train[ : index]), axis = 0)	
					batch_loss, batch_accuracy, _ = session.run([self.loss, self.accuracy, self.train_operation], 
										    feed_dict = {self.X: X_remainder, 
										    		 self.y: y_remainder,
										    		 self.keep_prob0: 0.75,
												 self.keep_prob1: 0.5})
					training_loss += batch_loss
					training_accuracy += batch_accuracy
					num_batches += 1
				
				training_loss /= num_batches
				training_accuracy /= num_batches
				
				num_validation_samples = X_validation.shape[0]
				predictions = np.empty(shape = (num_validation_samples, ), dtype = np.int32)
				for i in range(0, num_validation_samples, self.batch_size):
					j = min(i + self.batch_size, num_validation_samples)
					predictions[i : j] = session.run(self.result, 
									 feed_dict = {self.X: X_validation[i : min(i + self.batch_size, num_validation_samples)],
										      self.y: y_validation[i : min(i + self.batch_size, num_validation_samples)]
										     })
				validation_accuracy = np.mean(predictions == y_validation)	
				
				print('Training Accuracy:   {:8.5}\tTraining Loss: {:8.5}'.format(training_accuracy, training_loss))
				print('Validation Accuracy: {:8.5}'.format(validation_accuracy))
				
				if (validation_accuracy > best_accuracy):
					best_accuracy = validation_accuracy
					print('New best accuracy:   {:8.5}'.format(best_accuracy))
					self.saver.save(session, self.model_path)
			
	def predict(self, X):
		num_samples = X.shape[0]
		predictions = np.empty(shape = (num_samples, ), dtype = np.int32)
		
		with tf.Session(graph = self.graph) as session:
			self.saver.restore(session, self.model_path)
			
			for i in range(0, num_samples, self.batch_size):
				predictions[i : min(i + self.batch_size, num_samples)] = session.run(self.result, feed_dict = {self.X: X[i : min(i + self.batch_size, num_samples)]})
		
		return predictions
	
	def measure_accuracy(self, X, y):
		return np.mean(self.predict(X) == y)
	
