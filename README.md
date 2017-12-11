MNIST Classifier
=====================================================
This repository contains a simple convolutional neural-network classifier for the mnist-dataset.

- The algorithm uses *2 convolutional-* and *2 max-pooling-layers* with a final *fully conncted* layer.
- The network is trained using *softmax-cross-entropy* as loss function and an *Adamoptimizer* with a set learning rate of *0.001*.
- It reaches an accuracy of 97.96% after 2 epochs of training (1 epoch being the full dataset)

Contact me if you have any questions or want to use the code.

Different stable-versions of the algorithm can be found in different commits,
e.g. ``git checkout 398690e`` checks out the first working version using an MLP instead of a CNN.
Go back to newest commit using ``git checkout master``.

Prerequisites
--------------
- Python 3.5.2 or newer

Packages
-------------
All packages can be installed using pip
- tensorflow 1.3.0
- numpy 1.13.1

Run Locally
-----------
- Clone the repo
- Run ``python3 main.py``
- The code will download the mnist-dataset if none is provided

Some hyperparameters can be set using parameters:
``python3 main.py --batchsize 10``

A list of all hyperparameters and their use can be found using:
``python3 main.py --help``
