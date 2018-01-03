# This file is the main entry point of the program
# python3 main.py --do_training=0 --load_path=./logs/model
import argparse
import tensorflow as tf
import sys
import mnist_input
import time
import model
import numpy as np
import mnist_helper
import kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)

# Create everything and start training
def run(_):
    input_pl, labels_pl, keep_pl = model.placeholders(28, 28, 2, FLAGS.classes)
    logits, weights = model.inference(input_pl, FLAGS.classes, keep_pl)
    loss = model.loss(logits, labels_pl, weights, .0005)
    train_op = model.train(loss, 0.5)
    accuracy = model.accuracy(logits, labels_pl)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    epoch_counter = 0

    saver = tf.train.Saver()

    #Load trained model if provided
    if FLAGS.load_path != '':
        print('[INFO] - Loading graph from: ', FLAGS.load_path)
        saver.restore(sess, FLAGS.load_path)


    t = int(round(time.time()))
    # Train
    if FLAGS.do_training != 0:
        for epoch in range(2):
            for batch in mnist_input.read(FLAGS.path, FLAGS.batchsize, isTraining=True):
                labels, imagePairs = mnist_input.get_similarity_batch(batch)
                feed_dict = {
                    input_pl: imagePairs,
                    labels_pl: np.array(labels).reshape(-1,1),
                    keep_pl: 0.4
                }
                output, total_loss, _ = sess.run([logits, loss, train_op], feed_dict=feed_dict)
                print('[INFO] Epoch: ', epoch, '\n',
                        'Prediction:\t ', np.array(output).reshape(-1), '\n',
                        'Labels:\t ', labels, '\n',
                        'Loss: ', total_loss, '\n')
        print('[INFO] - Training time in minutes: ', (int(round(time.time())) - t)/60)

    # Save model after training
    save_path = saver.save(sess, FLAGS.log_dir + "/model.ckpt")
    print("[INFO] - Model saved in file: ", save_path)
    # Test
    for batch in mnist_input.read(FLAGS.path, 0, isTraining=False):
        labels, imagePairs = mnist_input.get_similarity_batch(batch)
        feed_dict = {
            input_pl: imagePairs,
            labels_pl: np.array(labels).reshape(-1,1),
            keep_pl: 1.0
        }
        acc = sess.run(accuracy, feed_dict=feed_dict)
        print('[INFO] Testing: \n',
                'Accuracy: ', acc, '\n')

    # Plotting Convolutions
    # for batch in mnist_input.read(FLAGS.path, 1, isTraining=False):
    #     if np.random.rand() < 0.1:
    #         _, images = batch
    #         feed_dict = {
    #             input_pl: images.astype(np.float32),
    #             keep_pl: 1.0
    #         }
    #         result1, result2 = sess.run([conv1,conv2], feed_dict=feed_dict)
    #         mnist_helper.show_conv_results(result1)
    #         mnist_helper.show_conv_results(result2)
    #         break

    # KMeans on test data
    for batch in mnist_input.read(FLAGS.path, 1000, isTraining=False):
        labels, images = batch
        # if np.random.rand() > 0.1:
        #     continue
        feed_dict = {
            input_pl: images.astype(np.float32),
            keep_pl: 1.0
        }
        features = sess.run(fc, feed_dict=feed_dict)
        features_std = StandardScaler().fit_transform(features)
        sklearn_pca = sklearnPCA(n_components=2)
        Z = sklearn_pca.fit_transform(features_std)
        k = 10
        centroids, cluster_labels = kmeans.run(k, Z)
        print('c.labels: ', cluster_labels)
        print('r.labels: ', labels)

        cluster_idxs = [np.where(np.array(cluster_labels) == i) for i in range(k)]
        cluster_idxs = np.array([cluster_idxs[i][0][0:10] for i in range(len(cluster_idxs))])
        print(np.array(cluster_idxs).shape, cluster_idxs)

        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
        np.random.shuffle(names)

        fig, ax = plt.subplots()
        images = np.array(images).reshape((-1,28,28))
        for i in range(k):
            cluster_data = [Z[j] for j in cluster_idxs[i]]
            cluster_data = np.array(cluster_data).reshape(10, 2)
            print(np.array(cluster_data).shape, cluster_data)
            imgs = [images[j] for j in cluster_idxs[i]]
            imgs = np.array(imgs).reshape(-1,28,28)
            x = cluster_data[:,0]
            y = cluster_data[:,1]


            for idx in range(len(imgs)):
                ax.scatter(x,y, c=names[i], s=0.5)
                ax.imshow(imgs[idx], extent=(x[idx],x[idx]+2,y[idx],y[idx]+2))

            x = centroids[i,0]
            y = centroids[i,1]
            plt.scatter(x, y, c=names[i], marker='^')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--path',
      type=str,
      default='data',
      help='Relative path to the mnist dataset'
    )
    parser.add_argument(
      '--batchsize',
      type=int,
      default='64',
      help='Number of images in one training batch'
    )
    parser.add_argument(
      '--classes',
      type=int,
      default='2',
      help='Number of classes'
    )
    parser.add_argument(
      '--load_path',
      type=str,
      default='',
      help='Relative path to a saved model'
    )
    parser.add_argument(
      '--log_dir',
      type=str,
      default='./logs',
      help='Relative path to the log files'
    )
    parser.add_argument(
      '--do_training',
      type=int,
      default=1,
      help='Whether or not the classifier should be trained or not'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
