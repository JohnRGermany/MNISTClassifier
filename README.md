MNIST Classifier
=====================================================
This repository contains a simple neural-network classifier for the mnist-dataset.

**CAUTION: THIS CODE IS NOT TESTED.**

The code was developed by myself in the simplest way possible.
No fine-tuning or prevention of overfitting was made.

- The algorithm uses a *3-layer MLP* with a final *softmax* layer as its classifier.
- The network is trained using *cross-entropy* as loss function and an *Adamoptimizer* with a set learning rate of 0.0001.
- It reaches an accuracy of over 90% after 3 epochs

Contact me if you have any questions or want to use the code.

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
