import tensorflow as tf

class Generator:

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

	def __init__(self, input_shape, output_shape, batch_size, model_path = '.', load = False):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.batch_size = batch_size
		self.model_path = model_path
		
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.X = tf.placeholder(tf.float32, shape = (None, 8, 8, 1) + input_shape, name = 'X')
			self.Y = tf.placeholder(tf.float32, shape = (None, 64, 64, 1) + output_shape, name = 'Y')
						
			self.loss = 
						
			self.train_operation =
				
	def train(self):
		pass
