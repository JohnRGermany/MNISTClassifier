import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def run(k, data):
    data_pl = tf.placeholder(tf.float32, shape=data.shape, name='Data_pl')
    kmeans = KMeans(inputs=data_pl, num_clusters=k, use_mini_batch=True)
    training_graph = kmeans.training_graph()
    (all_scores, cluster_idx, scores, cluster_centers_initialized, cluster_centers_var, init_op, train_op) = training_graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(init_op, feed_dict={data_pl: data})
    idx = []
    while True:
        idx_old = idx
        _, idx = sess.run([train_op, cluster_idx], feed_dict={data_pl: data})
        if np.array_equal(idx_old, idx):
            break

    cluster_labels = sess.run(cluster_idx, feed_dict={data_pl: data})[0]
    centroids = cluster_centers_var.eval(sess)

    return centroids, cluster_labels
