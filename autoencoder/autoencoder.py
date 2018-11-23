class Autoencoder:
	
	def __init__(self, input_shape, model_path, batch_size, first_run = True):
		self.image_height, self.image_width, self.image_num_channels = input_shape
		self.model_path = model_path
		self.batch_size = batch_size
		self.num_classes = num_classes
		
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.X = tf.placeholder(tf.float32, shape = (None, self.image_height, self.image_width, self.image_num_channels), name = 'X')
			
			self.keep_prob0 = tf.placeholder_with_default(1.0, shape = ())
			self.keep_prob1 = tf.placeholder_with_default(1.0, shape = ())
			
			self.output = tf.layers.conv2d(self.X, filters = 8, kernel_size = (5, 5), strides = (1, 1), padding = 'same', activation = tf.nn.relu)
			self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'same')
			self.output = tf.nn.dropout(self.output, self.keep_prob0)			
			print(self.output.shape)
			
			self.output = tf.layers.conv2d(self.output, filters = 16, kernel_size = (5, 5), strides = (1, 1), padding = 'same', activation = tf.nn.relu)
			self.output = tf.layers.max_pooling2d(self.output, pool_size = (2, 2), strides = (2, 2), padding = 'same')
			self.output = tf.nn.dropout(self.output, self.keep_prob0)
			print(self.output.shape)
			
			self.output = tf.layers.conv2d_transpose(self.output, filters = 16, kernel_size = (5, 5), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
			self.output = tf.nn.dropout(self.output, self.keep_prob0)			
			print(self.output.shape)
			
			self.output = tf.layers.conv2d_transpose(self.output, filters = 8, kernel_size = (5, 5), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
			self.output = tf.nn.dropout(self.output, self.keep_prob0)
			print(self.output.shape)
			
			self.loss = tf.reduce_sum((self.output - self.X) ** 2))
			
			self.train_operation = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(self.loss)
			self.saver = tf.train.Saver()
			
			if first_run:
				with tf.Session(graph = self.graph) as session:
					session.run(tf.global_variables_initializer())
					self.saver.save(session, self.model_path)
	
	def train(self):
	
	def predict(self):
