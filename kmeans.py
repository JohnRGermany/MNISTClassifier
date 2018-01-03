import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def run(k, X_pl, Y_pl, distance, sess):

    centroids = init_cluster_centroids()
    cluster_label = tf.argmin(distance, 0)

    while True:
        cluter_idxs = sess.run(cluster_labels, feed_dict={X_pl: data, Y_pl: centroids})
        centroids = sess.run(recompute_centroids())
    return centroids

def init_cluster_centroids(data, k):
    return data[:k]

# def recompute_centroids():
