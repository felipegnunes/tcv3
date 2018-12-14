import numpy as np
import tensorflow as tf
import cv2
import sys

import dataset_manip
import image_manip

# CONSTANTS

BATCH_SIZE = 64

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

### LOADING IMAGES ###

DATASET_DIRECTORY = '../data_part1'

X, _, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
X = np.concatenate((X, X_hidden), axis = 0)

resized_images =  np.empty(shape = (X.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype = np.float32)
for i in range(X.shape[0]):
	resized_images[i] = cv2.resize(X[i], (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_AREA).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)

X = resized_images

X -= 0.5
X /= 0.5

### LOADING MNIST IMAGES ###

#mnist = tf.keras.datasets.mnist
#(X_train, _), (X_test, _) = mnist.load_data()
#X_mnist = np.concatenate((X_train, X_test))
#X_mnist = X_mnist.astype(np.float32)
#X_mnist /= 255.0

#resized_images = np.empty(shape = (X_mnist.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype = np.float32)
#for i in range(X_mnist.shape[0]):
#	resized_images[i] = cv2.resize(X_mnist[i], (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_LINEAR).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)

#X = np.concatenate((X, resized_images))

print('X.shape = ' + str(X.shape))

#cv2.imshow('image', X[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

### PLACEHOLDERS ###

z = tf.placeholder(tf.float32, shape = (None, 64), name = 'z')
X_real = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, 1), name = 'X_real')
is_training = tf.placeholder(tf.bool, name = 'is_training')

### DISCRIMINATOR ###

def discriminator(X, reuse = None):
	with tf.variable_scope('discriminator', reuse = reuse):
		print(X.shape)
		conv1 = tf.layers.conv2d(inputs = X, filters = 16, kernel_size = (4, 4), strides = (2, 2), padding = 'same')
		conv1 = tf.nn.leaky_relu(features = conv1, alpha = 0.2)
		print(conv1.shape)
		
		conv2 = tf.layers.conv2d(inputs = conv1, filters = 32, kernel_size = (4, 4), strides = (2, 2), padding = 'same')
		conv2 = tf.nn.leaky_relu(features = tf.layers.batch_normalization(conv2, training = is_training), alpha = 0.2)
		print(conv2.shape)
		
		conv3 = tf.layers.conv2d(inputs = conv2, filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same')
		conv3 = tf.nn.leaky_relu(features = tf.layers.batch_normalization(conv3, training = is_training), alpha = 0.2)
		print(conv3.shape)
		
		conv4 = tf.layers.conv2d(inputs = conv3, filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'same')
		conv4 = tf.nn.leaky_relu(features = tf.layers.batch_normalization(conv4, training = is_training), alpha = 0.2)
		print(conv4.shape)
		
		conv5 = tf.layers.conv2d(inputs = conv4, filters = 1, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')
		conv5 = tf.reshape(tensor = conv5, shape = (-1, 1))
		print(conv5.shape)
		print()
		
		return conv5
		
### GENERATOR ###

def generator(z, reuse = None):
	with tf.variable_scope('generator', reuse = reuse):	
		print(z.shape)
		conv1 = tf.reshape(tensor = z, shape = (-1, 1, 1, 64))
		conv1 = tf.layers.conv2d_transpose(inputs = conv1, filters = 128, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')
		conv1 = tf.nn.leaky_relu(features = tf.layers.batch_normalization(conv1, training = is_training), alpha = 0.2)
		print(conv1.shape)
		
		conv2 = tf.layers.conv2d_transpose(inputs = conv1, filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same')
		conv2 = tf.nn.leaky_relu(features = tf.layers.batch_normalization(conv2, training = is_training), alpha = 0.2)
		print(conv2.shape)
		
		conv3 = tf.layers.conv2d_transpose(inputs = conv2, filters = 32, kernel_size = (4, 4), strides = (2, 2), padding = 'same')
		conv3 = tf.nn.leaky_relu(features = tf.layers.batch_normalization(conv3, training = is_training), alpha = 0.2)
		print(conv3.shape)
		
		conv4 = tf.layers.conv2d_transpose(inputs = conv3, filters = 16, kernel_size = (4, 4), strides = (2, 2), padding = 'same')
		conv4 = tf.nn.leaky_relu(features = tf.layers.batch_normalization(conv4, training = is_training), alpha = 0.2)
		print(conv4.shape)
		
		conv5 = tf.layers.conv2d_transpose(inputs = conv4, filters = 1, kernel_size = (4, 4), strides = (2, 2), padding = 'same')
		conv5 = tf.nn.tanh(conv5)
		print(conv5.shape)
		print()
		
		return conv5

### DEFINING MODEL ###

G = generator(z)
D_fake = discriminator(G)

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.ones_like(D_fake)))

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
	G_trainer = tf.train.AdamOptimizer(5e-4, beta1 = 0.5).minimize(G_loss, var_list = g_vars)

D_real = discriminator(X_real, reuse = True)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real, labels = tf.ones_like(D_real) * 0.9))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.zeros_like(D_fake)))

D_loss = D_loss_real + D_loss_fake

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):	
	d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]	
	D_trainer = tf.train.AdamOptimizer(2e-4, beta1 = 0.5).minimize(D_loss, var_list = d_vars)

### TRAINING ###

sess = tf.Session()
sess.run(tf.global_variables_initializer())

NUM_SAMPLES = X.shape[0]

NUM_ITERATIONS = 10000
iteration = 1

while iteration <= NUM_ITERATIONS:

	for i in range(0, NUM_SAMPLES, BATCH_SIZE):
		print('Iteration {}'.format(iteration))		
		j = min(i + BATCH_SIZE, NUM_SAMPLES)
		
		z_batch = np.random.normal(0, 1, size = (BATCH_SIZE, 64))
		X_batch = X[i : j]
		
		_, loss_dis = sess.run([D_trainer, D_loss], feed_dict = {X_real: X_batch, z: z_batch, is_training: True})
		
		z_batch = np.random.normal(0, 1, size = (BATCH_SIZE, 64))
		_, loss_gen = sess.run([G_trainer, G_loss], feed_dict = {z: z_batch, is_training: True})
		z_batch = np.random.normal(0, 1, size = (BATCH_SIZE, 64))
		_, loss_gen = sess.run([G_trainer, G_loss], feed_dict = {z: z_batch, is_training: True})				
						
		print('Loss Discriminator: {}'.format(loss_dis))
		print('Loss Generator: {}'.format(loss_gen))
		
		if iteration % 10 == 0:
			### PROGRESS VISUALIZATION ###
			
			z_samples = np.random.uniform(0, 1, size = (25, 64))
			gen_samples = sess.run(G, feed_dict = {z: z_samples, is_training: False})

			gen_samples -= np.amin(gen_samples)
			gen_samples /= np.amax(gen_samples)
			gen_samples *= 255

			images = np.empty(shape = (IMAGE_HEIGHT * 5, 0, 1))
			for i in range(5):
				imgs_column = np.empty(shape = (IMAGE_HEIGHT * 5, IMAGE_WIDTH, 1))
				for j in range(5):
					imgs_column[j * IMAGE_HEIGHT : (j + 1) * IMAGE_HEIGHT][:] = gen_samples[i * 5 + j]
				images = np.concatenate((images, imgs_column), axis = 1)
			cv2.imwrite('/home/felipe/tcv3/trabalho5/results/iteration_' + str(iteration) + '.png', images)	
				
			### SAVING ###

			saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'generator'))
			saver.save(sess, './model_files/generator')
		
		iteration += 1
		if iteration > NUM_ITERATIONS:
			break
			
	permutation = np.random.permutation(NUM_SAMPLES)
	X = X[permutation]
	
