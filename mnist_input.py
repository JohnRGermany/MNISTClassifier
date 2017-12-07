# This file contains functions to read and process input images

import tensorflow as tf
import os
import numpy as np
import struct
import gzip

from six.moves import urllib

IMAGES_NAME = 'train-images-idx3-ubyte'
LABELS_NAME = 'train-labels-idx1-ubyte'
DOWNLOAD_URL = 'http://yann.lecun.com/exdb/mnist/'

# Returns an iterator containing #-batchSize (label, image)-pairs
def read(path, batchSize):
    download(path, IMAGES_NAME, DOWNLOAD_URL)
    download(path, LABELS_NAME, DOWNLOAD_URL)

    imagesPath = os.path.join(path, IMAGES_NAME)
    labelsPath = os.path.join(path, LABELS_NAME)

    #TODO: stream each image rather than loading everything into memory

    with open(imagesPath, 'rb') as imagesFile:
        # TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
        # [offset] [type]          [value]          [description]
        # 0000     32 bit integer  0x00000803(2051) magic number
        # 0004     32 bit integer  60000            number of images
        # 0008     32 bit integer  28               number of rows
        # 0012     32 bit integer  28               number of columns
        # 0016     unsigned byte   ??               pixel
        # 0017     unsigned byte   ??               pixel
        # ........
        # xxxx     unsigned byte   ??               pixel
        # '>' means big-endian, each 'i' is one int (4 bytes each)
        magic, num, rows, cols = struct.unpack('>iiii', imagesFile.read(16))
        assert magic == 2051
        images = np.fromfile(imagesFile, dtype=np.uint8).reshape(num, rows * cols)

    with open(labelsPath, 'rb') as labelsFile:
        # TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
        # [offset] [type]          [value]          [description]
        # 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        # 0004     32 bit integer  60000            number of items
        # 0008     unsigned byte   ??               label
        # 0009     unsigned byte   ??               label
        # ........
        # xxxx     unsigned byte   ??               label
        # '>' means big-endian, each 'i' is one int (4 bytes each)
        magic, num = struct.unpack('>ii', labelsFile.read(8))
        assert magic == 2049
        labels = np.fromfile(labelsFile, dtype=np.uint8)

    imageBatch = lambda i: (labels[i:i+batchSize], images[i:i+batchSize])

    for i in range(0, len(labels), batchSize):
        yield imageBatch(i)

# Download the files if they are not already there
def download(dataDir, filename, url):
    if not os.path.isdir(dataDir):
        os.makedirs(dataDir)
    path = os.path.join(dataDir, filename)
    url = os.path.join(url, filename) + '.gz'
    if not tf.gfile.Exists(path):
        dlpath, headers = urllib.request.urlretrieve(url, path + '.gz')
        print('[INFO] Downloaded: ', filename + '.gz', '\n'
            'path: ', dlpath, '\n'
            'Headers: ', headers)
        with gzip.GzipFile(path + ".gz", 'rb') as inF:
            with open(path, 'wb') as outF:
                s = inF.read()
                outF.write(s)
                print('[INFO] Extracted: ', dlpath, '\n'
                    'into: ', path, '\n')
    assert tf.gfile.Exists(path)
