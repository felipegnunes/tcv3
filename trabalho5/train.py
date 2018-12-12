import numpy as np
import tensorflow as tf
import cv2
import sys

import dataset_manip

# CONSTANTS

BATCH_SIZE = 128

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

### LOADING IMAGES ###

DATASET_DIRECTORY = '../data_part1'

X, _, X_hidden = dataset_manip.load_dataset(DATASET_DIRECTORY)
X = np.concatenate((X, X_hidden), axis = 0)

resized_images =  np.empty(shape = (X.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype = np.float32)
for i in range(X.shape[0]):
	resized_images[i] = cv2.resize(X[i], (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation = cv2.INTER_AREA).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)

X = resized_images

print('X.shape = ' + str(X.shape))

#cv2.imshow('image', X[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

### PLACEHOLDERS ###

z = tf.placeholder(tf.float32, shape = (None, 64), name = 'z')
X_real = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, 1), name = 'X_real')
is_training = tf.placeholder(tf.bool, name = 'is_training');

### DISCRIMINATOR ###

def discriminator(X, reuse = None):
	with tf.variable_scope('discriminator', reuse = reuse):
		fc1 = tf.reshape(tensor = X, shape = (-1, 1024))
		fc1 = tf.layers.dense(inputs = fc1, units = 256, use_bias = True)
		fc1 = tf.nn.leaky_relu(features = fc1, alpha = 0.01)
		
		fc2 = tf.layers.dense(inputs = fc1, units = 64, use_bias = True)
		fc2 = tf.nn.leaky_relu(features = fc2, alpha = 0.01)
		
		fc3 = tf.layers.dense(inputs = fc2, units = 1)
		print(fc3.shape)
		print()
		
		return fc3
		
		#conv1 = tf.layers.conv2d(inputs = X, filters = 4, kernel_size = (3, 3), strides = (2, 2), padding = 'same') 
		#conv1 = tf.layers.batch_normalization(conv1, training = is_training)
		#conv1 = tf.nn.leaky_relu(features = conv1, alpha = 0.01)
		
		#conv2 = tf.layers.conv2d(inputs = conv1, filters = 8, kernel_size = (3, 3), strides = (2, 2), padding = 'same', use_bias = False)
		#conv2 = tf.nn.leaky_relu(features = conv2, alpha = 0.01)
		
		#conv3 = tf.layers.conv2d(inputs = conv2, filters = 16, kernel_size = (3, 3), strides = (2, 2), padding = 'same')
		#conv3 = tf.nn.leaky_relu(features = conv3, alpha = 0.01)
		
		#fc1 = tf.reshape(tensor = conv3, shape = (-1, conv2.shape[1] * conv2.shape[2] * conv2.shape[3]))
		#fc1 = tf.layers.dense(inputs = fc1, units = 1)
		
		#return fc1

### GENERATOR ###

def generator(z, reuse = None):
	with tf.variable_scope('generator', reuse = reuse):	
		fc1 = tf.layers.dense(inputs = z, units = 256, use_bias = False)
		#fc1 = tf.layers.batch_normalization(fc1, training = is_training)
		fc1 = tf.nn.leaky_relu(features = fc1, alpha = 0.01)
		print(fc1.shape)
		
		fc2 = tf.layers.dense(inputs = fc1, units = 512, use_bias = False)
		#fc2 = tf.layers.batch_normalization(fc2, training = is_training)
		fc2 = tf.nn.leaky_relu(features = fc2, alpha = 0.01)
		print(fc2.shape)
		
		fc3 = tf.layers.dense(inputs = fc2, units = 1024, use_bias = False)
		#fc3 = tf.layers.batch_normalization(fc3, training = is_training)
		fc3 = tf.reshape(tensor = fc3, shape = (-1, 32, 32, 1))
		print(fc3.shape)
		print()
		
		return tf.sigmoid(fc3)

### DEFINING MODEL ###

G = generator(z)
D_fake = discriminator(G)
D_real = discriminator(X_real, reuse = True)

### LOSSES ###

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real, labels = tf.ones_like(D_real) * 0.9))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.zeros_like(D_fake)))

D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.ones_like(D_fake)))

### TRAINERS ###

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'discriminator' in var.name]
g_vars = [var for var in tvars if 'generator' in var.name]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):	
	D_trainer = tf.train.AdamOptimizer(5e-4).minimize(D_loss, var_list = d_vars)
	G_trainer = tf.train.AdamOptimizer(1e-2).minimize(G_loss, var_list = g_vars)

### TRAINING ###

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	NUM_SAMPLES = X.shape[0]

	for epoch in range(1, 5000):
		print('Epoch {}'.format(epoch))
		
		for i in range(0, NUM_SAMPLES, BATCH_SIZE):
				j = min(i + BATCH_SIZE, NUM_SAMPLES)
				
				z_batch = np.array([np.random.normal(0, 1, size = (64, )) for k in range(BATCH_SIZE)])
				X_batch = X[i : j]
				
				_, loss_dis = sess.run([D_trainer, D_loss], feed_dict = {X_real: X_batch, z: z_batch, is_training: True})
				
				z_batch = np.array([np.random.normal(0, 1, size = (64, )) for k in range(BATCH_SIZE)])
				_, loss_gen = sess.run([G_trainer, G_loss], feed_dict = {z: z_batch, is_training: True})
				
				print('Loss Discriminator: {}'.format(loss_dis))
				print('Loss Generator: {}'.format(loss_gen))
		
		permutation = np.random.permutation(NUM_SAMPLES)
		X = X[permutation]
		
		sample_z = np.random.uniform(0, 1, size = (1, 64))
		gen_sample = sess.run(G, feed_dict = {z: sample_z, is_training: False}).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
		
		#gen_sample *= 255
		#print(gen_sample.shape)
		#print(np.amin(gen_sample))
		#print(np.amax(gen_sample))
		
		#gen_sample -= np.amin(gen_sample)
		#gen_sample /= np.amax(gen_sample)
		gen_sample *= 255
		
		cv2.imwrite('/home/felipe/tcv3/trabalho5/results/' + str(epoch) + '.png', gen_sample)	
	
### SAVING ###



