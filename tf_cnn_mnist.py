import math
import random
import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials import mnist 

#load MNIST data, store in tmp/data dir
digits = mnist.input_data.read_data_sets("/tmp/data", one_hot=True)

#print shape of training data
print ("Number of images (training) : ",len(digits.train.images))
print ("Number of images (testing) : ", len(digits.test.images))

#build a 3 layer network. 768, 512, 256 nodes
hidden_1, hidden_2, hidden_3 = 768, 512, 256

#number of output classes, batch sizes
n_classes,batch_size = digits.train.labels.shape[1], 64

#define features and labels, and their data types. MNIST is a 28x28 image dataset ==> 784 pixels/values per image.
X,y=tf.placeholder('float',[None, 784]), tf.placeholder('float')

def conv2d(data, weights):
	return tf.nn.conv2d(data, weights, strides=[1,1,1,1], padding = 'SAME') #one pixel at a time, for all pixels

def maxpool2d(x):
	#ksize = size of window; strides = movement of window
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #2x2 pixels at a time. padding is for images that end in edges (like the batch_size)

def convolutional_neural_network(data):

	weights ={'w_conv_1' : tf.Variable(tf.random_normal([3,3,1,32])), #conv 3x3, 1 input, 32 nodes for ouput
			  'w_conv_2' : tf.Variable(tf.random_normal([3,3,32,64])), #number of inputs has changed now.
			  'w_full_c' : tf.Variable(tf.random_normal([7*7*64,1024])), 
			  'w_output' : tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv_1' : tf.Variable(tf.random_normal([32])),
			  'b_conv_2' : tf.Variable(tf.random_normal([64])),
			  'b_full_c' : tf.Variable(tf.random_normal([1024])),
			  'b_output' : tf.Variable(tf.random_normal([10])),}

	data = tf.reshape(data, shape=[-1, 28, 28, 1]) #give -1 and 1 as params, 28, 28 because they were the inital image sizes.

	c1 = conv2d(data, weights['w_conv_1'])
	c1 = maxpool2d(c1)
	c2 = conv2d(c1, weights['w_conv_2'])
	c2 = maxpool2d(c2)
	
	fc = tf.reshape(c2, [-1, 7*7*64])
	fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_full_c']), biases['b_full_c']))

	op_l = tf.add(tf.matmul(fc, weights['w_output']), biases['b_output'])

	return op_l

def train(x):

	#define output, cost function, optimizer, epochs, and metrics for the network
	prediction = convolutional_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	epochs = 10

	#run the network
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(int(digits.train.num_examples//batch_size)):
				epoch_x,epoch_y = digits.train.next_batch(batch_size)
				_,c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_loss+=c
			print("Epoch number : ", epoch,", completed out of ",epochs)
			print("Loss : ", epoch_loss)

		#compare predicted vs actual for samples.
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))

		#get accuracy
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print("Accuracy : ", accuracy.eval({x:digits.test.images, y:digits.test.labels}))

train(X)