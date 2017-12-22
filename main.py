# This file is the main entry point of the program

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
    input_pl, labels_pl, keep_pl = model.placeholders(28, 28, 1, FLAGS.classes)
    logits, conv1, conv2, fc = model.inference(input_pl, FLAGS.classes, keep_pl)
    loss = model.loss(logits, labels_pl)
    train_op = model.train(loss, .001)
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
                labels, images = batch
                oneHotLabel = np.zeros((FLAGS.batchsize, FLAGS.classes))
                oneHotLabel[np.arange(FLAGS.batchsize), labels] = 1
                feed_dict = {
                    input_pl: images.astype(np.float32),
                    labels_pl: oneHotLabel.astype(np.float32),
                    keep_pl: 0.4
                }
                logitsput, total_loss, _ = sess.run([logits, loss, train_op], feed_dict=feed_dict)
                print('[INFO] Epoch: ', epoch, '\n',
                        'Prediction:\t ', np.array(logitsput).argmax(axis=1), '\n',
                        'Labels:\t ', labels, '\n',
                        'Loss: ', total_loss, '\n')
        print('[INFO] - Training time in minutes: ', (int(round(time.time())) - t)/60)

    # Save model after training
    save_path = saver.save(sess, FLAGS.log_dir + "/model.ckpt")
    print("[INFO] - Model saved in file: ", save_path)
    # Test
    # for batch in mnist_input.read(FLAGS.path, 0, isTraining=False):
    #     labels, images = batch
    #     oneHotLabel = np.zeros((len(labels), FLAGS.classes))
    #     oneHotLabel[np.arange(len(labels)), labels] = 1
    #     feed_dict = {
    #         input_pl: images.astype(np.float32),
    #         labels_pl: oneHotLabel.astype(np.float32),
    #         keep_pl: 1.0
    #     }
    #     acc = sess.run(accuracy, feed_dict=feed_dict)
    #     print('[INFO] Testing: \n',
    #             'Accuracy: ', acc, '\n')
    #
    # # Plotting Convolutions
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
    for batch in mnist_input.read(FLAGS.path, 30, isTraining=False):
        labels, images = batch
        if np.random.rand() > 0.1:
            continue
        feed_dict = {
            input_pl: images.astype(np.float32),
            keep_pl: 1.0
        }
        features = sess.run(fc, feed_dict=feed_dict)
        features_std = StandardScaler().fit_transform(features)
        sklearn_pca = sklearnPCA(n_components=2)
        Z = sklearn_pca.fit_transform(features_std)

        k = 10
        centroids, cluster_labels = kmeans.run(k, features)

        cluster_idxs = [np.where(np.array(cluster_labels) == i) for i in range(k)]

        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
        np.random.shuffle(names)

        fig, ax = plt.subplots()
        for i in range(k):
            cluster_data = [Z[j] for j in cluster_idxs[i]][0]
            imgs = np.array(images).reshape((-1,28,28))
            imgs = [imgs[j] for j in cluster_idxs[i]]
            imgs = np.array(imgs).reshape(-1,28,28)
            print(np.array(imgs).shape)
            x = cluster_data[:,0]
            y = cluster_data[:,1]


            for idx in range(len(imgs)):
                ax.scatter(x,y, c=names[i])
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
      default='50',
      help='Number of images in one training batch'
    )
    parser.add_argument(
      '--classes',
      type=int,
      default='10',
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
