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
				
				validation_accuracy = self._measure_accuracy_online(session, X_validation, y_validation) #np.mean(predictions == y_validation)	
				
				print('Training Accuracy:   {:8.5}\tTraining Loss: {:8.5}'.format(training_accuracy, training_loss))
				print('Validation Accuracy: {:8.5}'.format(validation_accuracy))
				
				if (validation_accuracy > best_accuracy):
					best_accuracy = validation_accuracy
					print('New best accuracy:   {:8.5}'.format(best_accuracy))
					self.saver.save(session, self.model_path)
