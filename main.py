# This file is the main entry point of the program

import argparse
import tensorflow as tf
import sys
import mnist_input
import time
import model
import numpy as np
import mnist_helper

# Create everything and start training
def run(_):
    input_pl, labels_pl, keep_pl = model.placeholders(28, 28, 1, FLAGS.classes)
    logits, conv1, conv2 = model.inference(input_pl, FLAGS.classes, keep_pl)
    loss = model.loss(logits, labels_pl)
    train_op = model.train(loss, .001)
    accuracy = model.accuracy(logits, labels_pl)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    epoch_counter = 0

    t = int(round(time.time()))
    # Train
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
    # Test
    for batch in mnist_input.read(FLAGS.path, 0, isTraining=False):
        labels, images = batch
        oneHotLabel = np.zeros((len(labels), FLAGS.classes))
        oneHotLabel[np.arange(len(labels)), labels] = 1
        feed_dict = {
            input_pl: images.astype(np.float32),
            labels_pl: oneHotLabel.astype(np.float32),
            keep_pl: 1.0
        }
        acc = sess.run(accuracy, feed_dict=feed_dict)
        print('[INFO] Testing: \n',
                'Accuracy: ', acc, '\n')

    # Plotting Convolutions
    for batch in mnist_input.read(FLAGS.path, 1, isTraining=False):
        if np.random.rand() < 0.1:
            _, images = batch
            feed_dict = {
                input_pl: images.astype(np.float32),
                keep_pl: 1.0
            }
            result1, result2 = sess.run([conv1,conv2], feed_dict=feed_dict)
            mnist_helper.show_conv_results(result1)
            mnist_helper.show_conv_results(result2)
            break



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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
