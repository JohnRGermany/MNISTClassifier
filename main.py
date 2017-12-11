# This file is the main entry point of the program

import argparse
import tensorflow as tf
import sys
import mnist_input
import time
import model
import numpy as np

# Create everything and start training
def run(_):
    input_pl, labels_pl = model.placeholders(28, 28, 1, FLAGS.classes)
    logits = model.inference(input_pl, FLAGS.classes)
    loss = model.loss(logits, labels_pl)
    train_op = model.train(loss, .001)
    accuracy = model.accuracy(logits, labels_pl)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    epoch_counter = 0
    # Train
    for epoch in range(2):
        for batch in mnist_input.read(FLAGS.path, FLAGS.batchsize, isTraining=True):
            labels, images = batch
            oneHotLabel = np.zeros((FLAGS.batchsize, FLAGS.classes))
            oneHotLabel[np.arange(FLAGS.batchsize), labels] = 1
            feed_dict = {
                input_pl: images.astype(np.float32),
                labels_pl: oneHotLabel.astype(np.float32)
            }
            logitsput, total_loss, _ = sess.run([logits, loss, train_op], feed_dict=feed_dict)
            print('[INFO] Epoch: ', epoch, '\n',
                    'Prediction:\t ', np.array(logitsput).argmax(axis=1), '\n',
                    'Labels:\t ', labels, '\n',
                    'Loss: ', total_loss, '\n')
    # Test
    for batch in mnist_input.read(FLAGS.path, 0, isTraining=False):
        labels, images = batch
        oneHotLabel = np.zeros((len(labels), FLAGS.classes))
        oneHotLabel[np.arange(len(labels)), labels] = 1
        feed_dict = {
            input_pl: images.astype(np.float32),
            labels_pl: oneHotLabel.astype(np.float32)
        }
        acc = sess.run(accuracy, feed_dict=feed_dict)
        print('[INFO] Testing: \n',
                'Accuracy: ', acc, '\n')

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
