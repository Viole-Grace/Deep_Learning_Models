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

def neural_network(data):

	layer_1 = {'weights':tf.Variable(tf.random_normal([784, hidden_1])),
			   'biases':tf.Variable(tf.random_normal([hidden_1]))}
	layer_2 = {'weights':tf.Variable(tf.random_normal([hidden_1, hidden_2])),
			   'biases':tf.Variable(tf.random_normal([hidden_2]))}
	layer_3 = {'weights':tf.Variable(tf.random_normal([hidden_2, hidden_3])),
	   		   'biases':tf.Variable(tf.random_normal([hidden_3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([hidden_3, n_classes])),
    				'biases':tf.Variable(tf.random_normal([n_classes]))}

	# (data*weights)+ bias ==> activation : neuron
	l1 = tf.add(tf.matmul(data, layer_1['weights']), layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, layer_2['weights']), layer_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, layer_3['weights']), layer_3['biases'])
	l3 = tf.nn.relu(l3)

	op_l = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return op_l

def train(x):

	#define output, cost function, optimizer, epochs, and metrics for the network
	prediction = neural_network(x)
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