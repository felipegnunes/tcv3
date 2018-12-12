import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data('./mnist')
x_train, x_test = x_train/255.0, x_test/255.0

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#plt.imshow(x_train[12], cmap = 'Greys')
#plt.show()

def generator(z, reuse = None):
	with tf.variable_scope('gen', reuse = reuse):
		hidden1 = tf.layers.dense(inputs = z, units = 128)
		alpha = 0.01
		hidden1 = tf.maximum(alpha * hidden1, hidden1)
		
		hidden2 = tf.layers.dense(inputs = hidden1, units = 128)
		hidden2 = tf.maximum(alpha * hidden2, hidden2)
		
		output = tf.layers.dense(hidden2, units = 28*28, activation = tf.nn.tanh)
		
		return output
		
def discriminator(X, reuse = None):
	with tf.variable_scope('dis', reuse = reuse):
		hidden1 = tf.layers.dense(inputs = X, units = 128)
		alpha = 0.01
		hidden1 = tf.maximum(alpha * hidden1, hidden1)
		
		hidden2 = tf.layers.dense(inputs = hidden1, units = 128)
		hidden2 = tf.maximum(alpha * hidden2, hidden2)
		
		logits = tf.layers.dense(hidden2, units = 1)
		output = tf.sigmoid(logits)
		
		return output, logits
		
real_images = tf.placeholder(tf.float32, shape = (None, 784))
z = tf.placeholder(tf.float32, shape = (None, 100))	

G = generator(z)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse = True)

# LOSSES

def loss_function(logits_in, labels_in):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_in, labels = labels_in))
	
D_real_loss = loss_function(D_logits_real, tf.ones_like(D_logits_real) * 0.9)
D_fake_loss = loss_function(D_logits_fake, tf.zeros_like(D_logits_fake))

D_loss = D_real_loss + D_fake_loss

G_loss = loss_function(D_logits_fake, tf.ones_like(D_logits_fake))

learning_rate = 0.001

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list = d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list = g_vars)

batch_size = 100
epochs = 2000

init = tf.global_variables_initializer()
samples = []

with tf.Session() as sess:
	sess.run(init)
	
	for epoch in range(epochs):
		num_batches = x_train.shape[0] // batch_size
		num_samples = x_train.shape[0]
		for i in range(0, num_samples, batch_size):
			j = min(i + batch_size, num_samples)
			batch_images = x_train[i : j].reshape(j - i, 784)
			batch_images = (batch_images * 2) - 1
			batch_z = np.random.uniform(-1, 1, size = (j - i, 100))
			
			_ = sess.run(D_trainer, feed_dict = {real_images: batch_images, z: batch_z})
			_ = sess.run(G_trainer, feed_dict = {z: batch_z})
			
		print('Epoch {}'.format(epoch))
		
		sample_z = np.random.uniform(-1, 1, size = (1, 100))
		gen_sample = sess.run(generator(z, reuse = True), feed_dict = {z: sample_z})
		
		gen_sample = gen_sample.reshape(28, 28)
		gen_sample -= np.amin(gen_sample)
		gen_sample /= np.amax(gen_sample)
		gen_sample *= 255
		
		cv2.imwrite('/home/felipe/results/' + str(epoch) + '.png', gen_sample)
		
		#samples.append(gen_sample.reshape(28, 28))

