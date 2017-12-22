# This file contains the neural network model that will be trained

import tensorflow as tf
import numpy as np

# Neural network definition
def inference(inputImage, num_classes, keep_rate):
    # Create 32 convolutions with 5x5 patches
    conv1 = tf.layers.conv2d(inputs=inputImage, filters=32, kernel_size=[5, 5],
        padding="SAME", activation=tf.nn.relu)
    # Reduce by factor 2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Create 64 convolutions with 5x5 patches
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
        padding="SAME", activation=tf.nn.relu)
    # Reduce by factor 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Squash to be used by feed forward nn
    reshape = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(reshape, 1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=keep_rate)
    fc = tf.layers.dense(dense, 128, activation=tf.nn.relu)
    logits = tf.layers.dense(fc, num_classes)

    return logits, conv1, conv2, fc

# Loss function of the network
def loss(logits, labels):
    return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

# Training function of the network
def train(loss, lr):
    return tf.train.AdamOptimizer(lr).minimize(loss)

def placeholders(rows, cols, num, classes):
    keep_pl = tf.placeholder(tf.float32, name='Dropout_rate_pl')
    input_pl = tf.placeholder(tf.float32, shape=[None, rows, cols, num], name='Input_pl')
    labels_pl = tf.placeholder(tf.float32, shape=[None, classes], name='Labels_pl')
    return input_pl, labels_pl, keep_pl

def accuracy(output, labels):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
