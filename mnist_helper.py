import matplotlib.pyplot as plt
import numpy as np
import math

def show_conv_results(data, filename=None):
    plt.figure()
    rows, cols = find_rows_cols(np.shape(data)[3])
    print('[INFO] - Plotting: ', np.shape(data))
    for i in range(np.shape(data)[3]):
        img = data[0, : , : , i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def find_rows_cols(x):
    num = math.floor(math.sqrt(x))
    while x % num != 0:
        num -= 1
    return num, x/num

def plot(data, filename=None):
    plt.figure()
    print('[INFO] - Plotting: ', np.shape(data))
    for i in range(2):
        img = data[ : , : , i]
        plt.subplot(1, 2, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
