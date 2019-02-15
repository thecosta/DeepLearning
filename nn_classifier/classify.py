import sys
import time
import keras
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10


if __name__ == '__main__':
    print('########## Start time!! ##########')
    start_time = time.time()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    #Import data
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)


    #Create the model
    x = tf.placeholder(tf.float32, [None, 3072])
    y_ = tf.placeholder(tf.int64, [None])


    # Variables
    W = tf.Variable(tf.zeros([3072, 10]))
    b = tf.Variable(tf.zeros([10]))


    # Output
    y = tf.matmul(x, W) + b
    #y = tf.matmul(y, W) + b

    print('########## Loss & Optimizer ##########')
    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_, 10), logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


    print('########## Training ##########')
    # Train
    for i in range(50):
        s = np.arange(x_train.shape[0])
        np.random.shuffle(s)
        x_tr = x_train[s]
        y_tr = y_train[s]
        batch_xs = x_tr[:100]
        batch_ys = y_tr[:100]
        loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys})
        print(f'Train {i}/5000 ==========> {loss}')#, end='\r')

    print(f'y size: {y.get_shape()}')
    print(f'y_ size: {y_.get_shape()}')

    print(f'x_test: {x_test.shape}') # 1000,3072
    print(f'y_test: {y_test.shape[0]}') # 50000,

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
    print(f'-------- {time.time() - start_time} s')
