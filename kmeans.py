import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def run(k, data, sess):
    centroids = tf.get_variable(shape=[k, num_features],
        initializer=tf.random_normal_initializer(mean=500, stddev=5), name='Centroids')
    data_pl = tf.placeholder(tf.float32, shape=[None, num_features], name='Data_pl')

    # TODO: the cluster function is sooo slow, there has to be a better way
    cluster = [tf.argmin(tf.sqrt(tf.reduce_mean(tf.pow(centroids - tf.gather(data_pl,i),2), axis=1))) for i in range(len(data))]
    move_centroids = [centroids[i].assign(tf.reduce_mean(data_pl, axis=0)) for i in range(k)]
    sess.run(tf.global_variables_initializer())

    print(centroids.eval(sess))
    cluster_labels = sess.run(cluster, feed_dict={data_pl: data})
    assert len(cluster_labels) == len(data)

    while True:
        cluster_idxs = [np.where(np.array(cluster_labels) == i) for i in range(k)]
        np.reshape(cluster_idxs, (k,-1))

        for i in range(k):
            cluster_data = [data[j] for j in cluster_idxs[i]][0]
            sess.run(move_centroids[i], feed_dict={data_pl: cluster_data})

        new_cluster_labels = sess.run(cluster, feed_dict={data_pl: data})

        ##THIS IS PLOTTING-----
        colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for i in range(k):
            cluster_data = [data[j] for j in cluster_idxs[i]][0]
            x = cluster_data[:,0]
            y = cluster_data[:,1]
            plt.scatter(x, y, c=colors[i], marker='o', s=10)

        cs = centroids.eval(sess)
        print(cs)
        x = cs[:,0]
        y = cs[:,1]
        plt.scatter(x, y, c='b', marker='^', s=200)

        plt.show()
        plt.clf()
        ##--------------------------

        if new_cluster_labels != cluster_labels:
            cluster_labels = new_cluster_labels
        else:
            break
    return centroids.eval(sess), cluster_labels


k = 8
num_features = 2
data = np.random.rand(100, num_features) * 1000
x = data[:,0]
y = data[:,1]
plt.scatter(x, y, color = 'g')
plt.show()
sess = tf.Session()
centroids, cluster_labels = run(k, data, sess)

cluster_idxs = [np.where(np.array(cluster_labels) == i) for i in range(k)]
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for i in range(k):
    cluster_data = [data[j] for j in cluster_idxs[i]][0]
    x = cluster_data[:,0]
    y = cluster_data[:,1]
    plt.scatter(x, y, c=colors[i], marker='o', s=50)

x = centroids[:,0]
y = centroids[:,1]
plt.scatter(x, y, c='b', marker='^', s=200)

print("FINISHED")
plt.show()
plt.clf()
