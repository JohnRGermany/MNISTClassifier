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
    input_pl = tf.placeholder(tf.float32, shape=[FLAGS.batchsize, 784], name='Input_pl')
    out = model.inference(input_pl, FLAGS.classes)
    labels_pl = tf.placeholder(tf.float32, shape=[FLAGS.batchsize, FLAGS.classes], name='Labels_pl')
    loss = model.loss(out, labels_pl)
    train_op = model.train(loss, .0001)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    epoch_counter = 0
    for epoch in range(5):
        for batch in mnist_input.read(FLAGS.path, FLAGS.batchsize):
            labels, images = batch
            oneHotLabel = np.zeros((FLAGS.batchsize, FLAGS.classes))
            oneHotLabel[np.arange(FLAGS.batchsize), labels] = 1
            feed_dict = {
                input_pl: images.astype(np.float32),
                labels_pl: oneHotLabel.astype(np.float32)
            }
            output, total_loss, _ = sess.run([out, loss, train_op], feed_dict=feed_dict)
            print('[INFO] Epoch: ', epoch, '\n',
                    'Prediction:\t ', np.array(output).argmax(axis=1), '\n',
                    'Labels:\t ', labels, '\n',
                    'Loss: ', total_loss, '\n')

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
      default='20',
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
