import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

n1 = 64
n2 = 32

for j in range(20):

	# define placeholder for inputs to network
	xs = tf.placeholder(tf.float32, [None, 784])
	ys = tf.placeholder(tf.float32, [None, 10])	

	## fc1 layer ##
	W_fc1 = weight_variable([784, n1])
	b_fc1 = bias_variable([n1])
	h_fc1 = tf.nn.relu(tf.matmul(xs, W_fc1) + b_fc1)

	## fc3 layer ##
	W_fc3 = weight_variable([n1, n2])
	b_fc3 = bias_variable([n2])
	h_fc3 = tf.nn.relu(tf.matmul(h_fc1, W_fc3) + b_fc3)

	## fc2 layer ##
	W_fc2 = weight_variable([n2, 10])
	b_fc2 = bias_variable([10])
	prediction = tf.nn.softmax(tf.matmul(h_fc3, W_fc2) + b_fc2)


	# the error between prediction and real data
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

	sess = tf.Session()

	init = tf.global_variables_initializer()
	sess.run(init)

	for i in range(2000):
		batch_xs, batch_ys = mnist.train.next_batch(1000)
		sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
		#print(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys})," ",compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000])," ",compute_accuracy(mnist.train.images[:1000], mnist.train.labels[:1000]))
	if j==0:
		print("parameters, train loss, test loss, train accuracy, test accuracy")
	print(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]),
	sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}),
	sess.run(cross_entropy, feed_dict={xs: mnist.test.images[:1000], ys: mnist.test.labels[:1000]}),
	compute_accuracy(mnist.train.images[:1000], mnist.train.labels[:1000]),
	compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
	tf.reset_default_graph()
	n1 = int(n1*1.2)
	n2 = int(n2*1.2)