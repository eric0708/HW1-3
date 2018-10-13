import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 設定參數
learning_rate = 0.5
training_steps = 1000
for index in (8,15):
    batch_size = 2 ** index
    logs_path = 'TensorBoard/'
    n_features = x_train.shape[1]
    n_labels = y_train.shape[1]

    # 建立 Feeds
    with tf.name_scope('Inputs'):
        x = tf.placeholder(tf.float32, [None, n_features], name = 'Input_Data')
    with tf.name_scope('Labels'):
        y = tf.placeholder(tf.float32, [None, n_labels], name = 'Label_Data')

    # 建立 Variables
    with tf.name_scope('ModelParameters'):
        W = tf.Variable(tf.zeros([n_features, n_labels]), name = 'Weights')
        b = tf.Variable(tf.zeros([n_labels]), name = 'Bias')

    # 開始建構深度學習模型
    with tf.name_scope('Model'):
        # Softmax
        prediction = tf.nn.softmax(tf.matmul(x, W) + b)
    with tf.name_scope('CrossEntropy'):
        # Cross-entropy
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices = 1))
        tf.summary.scalar("Loss", loss)
    with tf.name_scope('GradientDescent'):
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', acc)
        var_grad=(loss,[x])


    # 初始化
    init = tf.global_variables_initializer()

    # 開始執行運算
    sess = tf.Session()
    sess.run(init)

    # 將視覺化輸出
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

    # 訓練
    for step in range(training_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
        
        if step % 50 == 0:
            print(sess.run(loss, feed_dict = {x: batch_xs, y: batch_ys}))
            print(sess.run(loss, feed_dict = {x: x_test, y: y_test}))
            print("train accuracy: ", sess.run(acc, feed_dict={x: x_train, y: y_train}))
            print("test accuracy: ", sess.run(acc, feed_dict={x: x_test, y: y_test}))
            
            
            
            
            
            summary = sess.run(merged, feed_dict = {x: batch_xs, y: batch_ys})
            writer.add_summary(summary, step)
    a=0
    b=0
    for step in range(training_steps):
        var_grad_val = sess.run(var_grad, feed_dict={x: x_train[step].reshape(1,784), 
                                                    y: y_train[step].reshape(1,10)})

    
        
        abc=list(var_grad_val)
        
        ooo = np.asarray(abc[0])
        a=a+(np.linalg.norm(ooo))

        
    tf.reset_default_graph()
    print(a)        


    print("---")
    # 準確率


    sess.close()