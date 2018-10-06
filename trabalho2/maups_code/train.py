# ---------------------------------------------------------------------------------------------------------- #
# Author: maups                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import numpy as np
import random
import time
import sys
import os

from data import load_multiclass_dataset, shuffle, split

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Parameters.                                                                                        #
# ---------------------------------------------------------------------------------------------------------- #
TRAIN_FOLDER = './train' # folder with training images
TEST_FOLDER = './test'   # folder with testing images
SPLIT_RATE = 0.8         # split rate for training and validation sets

IMAGE_HEIGHT = 77  # height of the image
IMAGE_WIDTH = 71   # width of the image
NUM_CHANNELS = 1   # number of channels of the image

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Load the training set, shuffle its images and then split them in training and validation subsets.  #
#         After that, load the testing set.                                                                  #
# ---------------------------------------------------------------------------------------------------------- #
X_train, y_train, classes_train = load_multiclass_dataset(TRAIN_FOLDER, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
X_train = X_train.reshape(-1, IMAGE_HEIGHT*IMAGE_WIDTH*NUM_CHANNELS)/255.
X_train, y_train = shuffle(X_train, y_train, seed=42)
X_train, y_train, X_val, y_val = split(X_train, y_train, SPLIT_RATE)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Create a training graph that receives a batch of images and their respective labels and run a      #
#         training iteration or an inference job. Train the last FC layer using fine_tuning_op or the entire #
#         network using full_backprop_op. A weight decay of 1e-4 is used for full_backprop_op only.          #
# ---------------------------------------------------------------------------------------------------------- #
graph = tf.Graph()
with graph.as_default():
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT*IMAGE_WIDTH*NUM_CHANNELS))
	y = tf.placeholder(tf.int64, shape = (None,))
	y_one_hot = tf.one_hot(y, len(classes_train))
	learning_rate = tf.placeholder(tf.float32)

	fc = tf.layers.dense(X, 512, activation=tf.nn.relu)
	out = tf.layers.dense(fc, len(classes_train), activation=tf.nn.sigmoid)

	loss = tf.reduce_mean(tf.reduce_sum((y_one_hot-out)**2))

	train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

	result = tf.argmax(out, 1)
	correct = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Run one training epoch using images in X_train and labels in y_train.                              #
# ---------------------------------------------------------------------------------------------------------- #
def training_epoch(session, op, lr):
	batch_list = np.random.permutation(len(X_train))

	start = time.time()
	train_loss = 0
	train_acc = 0
	for j in range(0, len(X_train), BATCH_SIZE):
		if j+BATCH_SIZE > len(X_train):
			break
		X_batch = X_train.take(batch_list[j:j+BATCH_SIZE], axis=0)
		y_batch = y_train.take(batch_list[j:j+BATCH_SIZE], axis=0)

		ret = session.run([op, loss, correct], feed_dict = {X: X_batch, y: y_batch, learning_rate: lr})
		train_loss += ret[1]*BATCH_SIZE
		train_acc += ret[2]

	pass_size = (len(X_train)-len(X_train)%BATCH_SIZE)
	print('Training Epoch:'+str(epoch)+' LR:'+str(lr)+' Time:'+str(time.time()-start)+' ACC:'+str(train_acc/pass_size)+' Loss:'+str(train_loss/pass_size))

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Evaluate images in Xv with labels in yv.                                                           #
# ---------------------------------------------------------------------------------------------------------- #
def evaluation(session, Xv, yv, name='Evaluation'):
	start = time.time()
	eval_loss = 0
	eval_acc = 0
	for j in range(0, len(Xv), BATCH_SIZE):
		ret = session.run([loss, correct], feed_dict = {X: Xv[j:j+BATCH_SIZE], y: yv[j:j+BATCH_SIZE]})
		eval_loss += ret[0]*min(BATCH_SIZE, len(Xv)-j)
		eval_acc += ret[1]

	print(name+' Epoch:'+str(epoch)+' Time:'+str(time.time()-start)+' ACC:'+str(eval_acc/len(Xv))+' Loss:'+str(eval_loss/len(Xv)))

	return eval_acc/len(Xv), eval_loss/len(Xv)

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Training loop.                                                                                     #
# ---------------------------------------------------------------------------------------------------------- #
NUM_EPOCHS_FULL = 50
S_LEARNING_RATE_FULL = 0.001
F_LEARNING_RATE_FULL = 0.0001
BATCH_SIZE = 64

with tf.Session(graph = graph) as session:
	# weight initialization
	session.run(tf.global_variables_initializer())

	# full optimization
	for epoch in range(NUM_EPOCHS_FULL):
		lr = (S_LEARNING_RATE_FULL*(NUM_EPOCHS_FULL-epoch-1)+F_LEARNING_RATE_FULL*epoch)/(NUM_EPOCHS_FULL-1)
		training_epoch(session, train_op, lr)

		val_acc, val_loss = evaluation(session, X_val, y_val, name='Validation')

