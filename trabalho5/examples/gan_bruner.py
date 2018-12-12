# https://github.com/jonbruner/generative-adversarial-networks/blob/master/gan-script.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

print('X_train: {}\nX_test: {}\ny_train: {}\ny_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

def discriminator(X, reuse = None):
	with tf.variable_scope('discriminator', reuse = reuse):
		conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = (5, 5), strides = (1, 1), padding = 'SAME')
		conv1 = tf.nn.leaky_relu(features = conv1, alpha = 0.01)
		conv1 = tf.layers.average_pooling2d(inputs = conv1, pool_size = (2, 2), strides = (2, 2), padding = 'SAME')

		conv2 = tf.layers.conv2d(inputs = conv1, filters = 64, kernel_size = (5, 5), strides = (1, 1), padding = 'SAME')
		conv2 = tf.nn.leaky_relu(features = conv2, alpha = 0.01)
		conv2 = tf.layers.average_pooling2d(inputs = conv2, pool_size = (2, 2), strides = (2, 2), padding = 'SAME')

		fc1 = tf.reshape(tensor = conv2, shape = (-1, conv2.shape[1] * conv2.shape[2] * conv2.shape[3]))
		fc1 = tf.layers.dense(inputs = fc1, units = 1024)
		fc1 = tf.nn.leaky_relu(features = fc1, alpha = 0.01)

		fc2 = tf.layers.dense(inputs = fc1, units = 1)

		return fc2 # fc2 gives logits
		
def generator(z, batch_size, reuse = None):
	with tf.variable_scope('generator', reuse = reuse):
		fc1 = tf.layers.dense(inputs = z, units = (56 * 56), use_bias = False)
		fc1 = tf.reshape(tensor = fc1, shape = (-1, 56, 56, 1))
		fc1 = tf.contrib.layers.batch_norm(inputs = fc1, epsilon = 1e-5)
		fc1 = tf.nn.leaky_relu(features = fc1, alpha = 0.01)

		conv1 = tf.layers.conv2d(inputs = fc1, filters = 50, kernel_size = (3, 3), strides = (2, 2), padding = 'SAME', use_bias = False)
		conv1 = tf.contrib.layers.batch_norm(inputs = conv1, epsilon = 1e-5)
		conv1 = tf.nn.leaky_relu(features = conv1, alpha = 0.01)
		
		conv2 = tf.layers.conv2d(inputs = conv1, filters = 25, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME', use_bias = False)
		conv2 = tf.contrib.layers.batch_norm(inputs = conv2, epsilon = 1e-5)
		conv2 = tf.nn.leaky_relu(features = conv2, alpha = 0.01)
		
		conv3 = tf.layers.conv2d(inputs = conv2, filters = 1, kernel_size = (1, 1), strides = (1, 1), padding = 'SAME', use_bias = False)
		conv3 = tf.sigmoid(conv3)
		
		return conv3
		
batch_size = 50

z = tf.placeholder(tf.float32, shape = (None, 100))
X = tf.placeholder(tf.float32, shape = (None, 28, 28, 1))

G = generator(z, batch_size)
D_real = discriminator(X)
D_fake = discriminator(G, reuse = True)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real, labels = tf.ones_like(D_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.zeros_like(D_fake)))

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.ones_like(D_fake)))

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'discriminator' in var.name]
g_vars = [var for var in tvars if 'generator' in var.name]
	
D_trainer_fake = tf.train.AdamOptimizer(1e-3).minimize(D_loss_fake, var_list = d_vars)
D_trainer_real = tf.train.AdamOptimizer(1e-3).minimize(D_loss_real, var_list = d_vars)
G_trainer = tf.train.AdamOptimizer(1e-3).minimize(G_loss, var_list = g_vars)

tf.summary.scalar('Generator_loss', G_loss)
tf.summary.scalar('Discriminator_loss_real', D_loss_real)
tf.summary.scalar('Discriminator_loss_fake', D_loss_fake)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

images_for_tensorboard = generator(z, batch_size, reuse = True)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# Pre-train discriminator
for i in range(300):
	z_batch = np.random.normal(0, 1, size = (batch_size, 100))
	_, _, dLossReal, dLossFake = sess.run([D_trainer_real, D_trainer_fake, D_loss_real, D_loss_fake], feed_dict = {X: X_train[ : batch_size].reshape(batch_size, 28, 28, 1), z: z_batch})
	
	z_batch = np.random.normal(0, 1, size = (batch_size, 100))
	summary = sess.run(merged, {z: z_batch, X: X_train[ : batch_size].reshape(batch_size, 28, 28, 1)})
	writer.add_summary(summary, i)
	
	print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

