import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction3, feed_dict={xs: v_xs, 
		W_fc13: W_fc1i, b_fc13: b_fc1i, W_fc23: W_fc2i, b_fc23: b_fc2i, W_fc33: W_fc3i, b_fc33: b_fc3i})
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

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.
ys = tf.placeholder(tf.float32, [None, 10])	

## fc1 layer ##
W_fc1 = weight_variable([784, 600])
b_fc1 = bias_variable([600])
h_fc1 = tf.nn.relu(tf.matmul(xs, W_fc1) + b_fc1)
W_fc12 = weight_variable([784, 600])
b_fc12 = bias_variable([600])
h_fc12 = tf.nn.relu(tf.matmul(xs, W_fc12) + b_fc12)
W_fc13 = tf.placeholder(tf.float32, [784, 600])
b_fc13 = tf.placeholder(tf.float32, [600])
h_fc13 = tf.nn.relu(tf.matmul(xs, W_fc13) + b_fc13)

## fc3 layer ##
W_fc3 = weight_variable([600, 400])
b_fc3 = bias_variable([400])
h_fc3 = tf.nn.relu(tf.matmul(h_fc1, W_fc3) + b_fc3)
W_fc32 = weight_variable([600, 400])
b_fc32 = bias_variable([400])
h_fc32 = tf.nn.relu(tf.matmul(h_fc12, W_fc32) + b_fc32)
W_fc33 = tf.placeholder(tf.float32, [600, 400])
b_fc33 = tf.placeholder(tf.float32, [400])
h_fc33 = tf.nn.relu(tf.matmul(h_fc13, W_fc33) + b_fc33)

## fc2 layer ##
W_fc2 = weight_variable([400, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc3, W_fc2) + b_fc2)
W_fc22 = weight_variable([400, 10])
b_fc22 = bias_variable([10])
prediction2 = tf.nn.softmax(tf.matmul(h_fc32, W_fc22) + b_fc22)
W_fc23 = tf.placeholder(tf.float32, [400, 10])
b_fc23 = tf.placeholder(tf.float32, [10])
prediction3 = tf.nn.softmax(tf.matmul(h_fc33, W_fc23) + b_fc23)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-8,1.0)), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction2,1e-8,1.0)), reduction_indices=[1]))
train_step2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy2)
cross_entropy3 = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction3,1e-8,1.0)), reduction_indices=[1]))

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(1024)
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
	#print(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys})," ",compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000])," ",compute_accuracy(mnist.train.images[:1000], mnist.train.labels[:1000]))
W_fc1save = sess.run(W_fc1)
b_fc1save = sess.run(b_fc1)
W_fc3save = sess.run(W_fc3)
b_fc3save = sess.run(b_fc3)
W_fc2save = sess.run(W_fc2)
b_fc2save = sess.run(b_fc2)
for j in range(8000):
	batch_xs2, batch_ys2 = mnist.train.next_batch(128)
	sess.run(train_step2, feed_dict={xs: batch_xs2, ys: batch_ys2})
W_fc12save = sess.run(W_fc12)
b_fc12save = sess.run(b_fc12)
W_fc32save = sess.run(W_fc32)
b_fc32save = sess.run(b_fc32)
W_fc22save = sess.run(W_fc22)
b_fc22save = sess.run(b_fc22)

a = -1.0
for m in range(300):
	W_fc1i = (1-a)*W_fc1save + a*W_fc12save
	b_fc1i = (1-a)*b_fc1save + a*b_fc12save
	W_fc2i = (1-a)*W_fc2save + a*W_fc22save
	b_fc2i = (1-a)*b_fc2save + a*b_fc22save
	W_fc3i = (1-a)*W_fc3save + a*W_fc32save
	b_fc3i = (1-a)*b_fc3save + a*b_fc32save
	
	print(a,
	sess.run(cross_entropy3, feed_dict={xs: mnist.train.images[:1000], ys: mnist.train.labels[:1000], 
		W_fc13: W_fc1i, b_fc13: b_fc1i, W_fc23: W_fc2i, b_fc23: b_fc2i, W_fc33: W_fc3i, b_fc33: b_fc3i }),
	sess.run(cross_entropy3, feed_dict={xs: mnist.test.images[:1000], ys: mnist.test.labels[:1000], 
		W_fc13: W_fc1i, b_fc13: b_fc1i, W_fc23: W_fc2i, b_fc23: b_fc2i, W_fc33: W_fc3i, b_fc33: b_fc3i }),
	compute_accuracy(mnist.train.images[:1000], mnist.train.labels[:1000]),
	compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
	a += 0.01
