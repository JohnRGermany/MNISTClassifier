# This file contains the neural network model that will be trained

import tensorflow as tf
import numpy as np

# Neural network definition
def inference(input_image, num_classes):
    net = tf.layers.dense(input_image, 512, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=.1))
    net = tf.layers.dense(net, 512, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=.1))
    net = tf.layers.dense(net, 128, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=.1))
    out = tf.nn.softmax(tf.layers.dense(net, num_classes, kernel_initializer=tf.random_normal_initializer(stddev=.001)))
    return out

# Loss function of the network
def loss(logits, labels):
    return tf.losses.softmax_cross_entropy(labels, logits)

# Training function of the network
def train(loss, lr):
    return tf.train.AdamOptimizer(lr).minimize(loss)
