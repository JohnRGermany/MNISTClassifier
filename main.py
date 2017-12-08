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
    input_pl = tf.placeholder(tf.float32, shape=[None, 784], name='Input_pl')
    out = model.inference(input_pl, FLAGS.classes)
    labels_pl = tf.placeholder(tf.float32, shape=[None, FLAGS.classes], name='Labels_pl')
    loss = model.loss(out, labels_pl)
    train_op = model.train(loss, .0001)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    epoch_counter = 0
    # Train
    for epoch in range(5):
        for batch in mnist_input.read(FLAGS.path, FLAGS.batchsize, isTraining=True):
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
    # Test
    t = f = 0
    for batch in mnist_input.read(FLAGS.path, 1, isTraining=False):
        label, image = batch
        feed_dict = {
            input_pl: image.astype(np.float32)
        }
        output = sess.run(out, feed_dict=feed_dict)
        prediction = np.array(output).argmax(axis=1).flatten()
        t += label[0] == prediction[0]
        f += 1-(label[0] == prediction[0])

        print('[INFO] Testing:\n',
                'Prediction:\t ', prediction[0], '\n',
                'Label:\t\t ', label[0], '\n',
                'Accuracy: ', t / (t+f), '\n')

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
