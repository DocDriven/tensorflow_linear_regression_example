# Copyright (c) 2018 DocDriven

'''This file implements a simple linear regression algorithm. The aim is to fit
a straight line of the form f(x) = ax + t that has some additional noise on it.
By default, the line has the function f(x) = 3x + 1. '''

## Libraries

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import shutil

## constants

LOGDIR = 'C:/Users/Sebastian/tensorflow/logs'

## linear regression implementation

def linear_regression(tr_input, tr_output, learning_rate, epochs=100):

    # remove old graph elements
    tf.reset_default_graph()
    
    # declare the variables for weight (w) and bias (b) and initialize them
    w = tf.Variable(0.0, dtype=tf.float32, name="weight")
    tf.summary.scalar("weight", w)
    b = tf.Variable(0.0, dtype=tf.float32, name="bias")
    tf.summary.scalar("bias", b)
    
    # declare placeholders for input and output
    X = tf.placeholder(dtype=tf.float32, name="input")
    Y = tf.placeholder(dtype=tf.float32, name="output")
    
    # define the linear model
    Y_hat = w * X + b
    
    # define the loss function (residual sum of squares or RSS)
    with tf.name_scope('loss'):
        loss = tf.square(Y - Y_hat)
        tf.summary.scalar('loss', loss)
    
    # define the optimizer
    with tf.name_scope('training'):
        optimizer = \
            tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # merge all summaries for the writer
    summ = tf.summary.merge_all()
    
    # create a session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # create file handler
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    
    # train for some epochs
    for epoch in range(epochs):
        for x, y in zip(tr_input, tr_output):
            sess.run(optimizer, feed_dict={X:x, Y:y})
            s = sess.run(summ, feed_dict={X:x, Y:y})
            writer.add_summary(s, epoch)
    
    # visualize the results
    w_opt, b_opt = sess.run([w, b])
    print("final weight: %f" % w_opt)
    print("final bias: %f" % b_opt)
    plt.plot(tr_input, tr_output, 'ro', tr_input, w_opt*tr_input+b_opt, 'bx')
    plt.show()
    
def main():
    
    # delete LOGDIR to start training from scratch
    shutil.rmtree(LOGDIR, ignore_errors=True)
    
    # generate training data
    tr_input = np.linspace(-5.0, 5.0)
    tr_output = 3*tr_input+1+np.random.randn(tr_input.shape[0])
    
    # do linear regression for different learning rates
    for learning_rate in [1e-3]:
        linear_regression(tr_input, tr_output, learning_rate)
    
    print("Training Done!")
    print("Run \'tensorboard --logdir=%s\' to see results!" % LOGDIR)
    
if __name__ == "__main__":
    main()