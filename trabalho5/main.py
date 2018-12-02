import tensorflow as tf

def main():
	graph = tf.Graph()
	
	with graph.as_default():
		X_noise = tf.placeholder(tf.float32, shape = (None, 8, 8, 1), name = 'X_noise')
		Y = tf.placeholder(tf.float32, shape = (None, 64, 64, 1), name = 'Y')
		
		
		# Gerador
		print(X_noise.shape)
		output = tf.layers.conv2d_transpose(X_noise, filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
		print(output.shape)
		output = tf.layers.conv2d_transpose(output, filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
		print(output.shape)
		output = tf.layers.conv2d_transpose(output, filters = 1, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = tf.nn.relu)		
		print(output.shape)				
		
		print('\nDiscriminador\n')
			
		# Discriminador
		X = tf.placeholder(tf.float32, shape = (None, 64, 64, 1), name = 'X')
		
		print(X.shape)
		output = tf.layers.conv2d_transpose(X, filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
		print(output.shape)
		output = tf.layers.conv2d_transpose(output, filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = tf.nn.relu)
		print(output.shape)
		output = tf.layers.conv2d_transpose(output, filters = 1, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = tf.nn.relu)		
		print(output.shape)			
						

if __name__ == '__main__':
	main()
	
X_noise = tf.placeholder(tf.float32, shape = (None, 8, 8, 1), name = 'X')	
