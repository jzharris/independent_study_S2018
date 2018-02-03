import os
import os.path as path

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

import matplotlib.pyplot as plt
from scipy.misc import toimage


def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def export_cifar10(category=0):
    print(x_train.shape)
    filter_train = None
    for i in range(y_train.shape[0]):
        if (i % 100 == 0):
            print('{} / {}'.format(i, y_train.shape[0]))
            # if (filter_train is not None):
            #     if (filter_train.shape[0] > 15):
            #         break
        if (y_train[i][category] == 1):
            if (filter_train is None):
                print(x_train[1].shape)
                filter_train = np.array(
                    np.reshape(x_train[i], (1, 32, 32, 3)))
            else:
                filter_train = np.append(filter_train,
                                         np.reshape(x_train[i], (1, 32, 32, 3)),
                                         axis=0)
    images = filter_train[:16]
    print(images.shape)
    # create a grid of 3x3 images
    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(toimage(images[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    if not path.exists('datasets/cifar10'):
        os.mkdir('datasets/cifar10')

    np.save('datasets/cifar10/set_{}.npy'.format(category), filter_train)


x_train, y_train, _, _ = load_cifar10()
export_cifar10(0)
export_cifar10(1)
export_cifar10(2)
export_cifar10(3)
export_cifar10(4)
export_cifar10(5)
export_cifar10(6)
export_cifar10(7)
export_cifar10(8)
export_cifar10(9)