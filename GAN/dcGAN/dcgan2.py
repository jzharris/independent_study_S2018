import os
import os.path as path

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt
from scipy.misc import toimage

from import_cifar10 import import_cifar10
from keras.models import model_from_json

# fix random seed for reproducibility
np.random.seed(7)


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4

        # In:  32 x 32 x 3, depth=1
        # Out: 16 x 16 x 3, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # In:  16 x 16 x 3, depth=64
        # Out: 8 x 8 x 3, depth=128
        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # In:  8 x 8 x 3, depth=128
        # Out: 4 x 4 x 3, depth=256
        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # In:  4 x 4 x 3, depth=256
        # Out: 4 x 4 x 3, depth=512
        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability, (4*4*512)=8192
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64 * 4
        dim = int(self.img_rows / 4)

        # In: 100
        # Out: dim x dim x depth = 8*8*256 = 16384
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In:  dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # In:  2*dim x 2*dim x depth/2
        # Out: 4*dim x 4*dim x depth/4
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # In:  4*dim x 4*dim x depth/4
        # Out: 8*dim x 8*dim x depth/8
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # In:  8*dim x 8*dim x depth/8
        # Out: (img_rows x img_rows x channel) grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(self.channel, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM


class CIFAR10_DCGAN(object):
    def __init__(self, category=0, load_weights=False):
        if not path.exists('out'):
            os.mkdir('out')

        if not path.exists('out/cifar10'):
            os.mkdir('out/cifar10')

        self.img_rows = 32
        self.img_cols = 32
        self.channel = 3
        self.category = category
        self.save_discr = "cifar10-discr-set_{}".format(self.category)
        self.save_adver = "cifar10-adver-set_{}".format(self.category)
        self.save_gener = "cifar10-gener-set_{}".format(self.category)

        self.x_train = import_cifar10(category=self.category)#[-1000:]
        print(self.x_train.shape)

        self.DCGAN = DCGAN(img_rows=self.img_rows, img_cols=self.img_cols, channel=self.channel)
        self.discriminator = self.load_weights(self.DCGAN.discriminator_model(), self.save_discr)
        self.adversarial = self.load_weights(self.DCGAN.adversarial_model(), self.save_adver)
        self.generator = self.load_weights(self.DCGAN.generator(), self.save_gener)

    def load_cifar10(self):
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

    def train(self, epochs=2000, batch_size=256, save_interval=0):
        batch_num = int(self.x_train.shape[0] / batch_size)
        location = 'weights/{}_weights.h5'.format(self.save_discr)
        if not path.exists(location):
            # Pre-train the discriminator network
            noise = np.random.uniform(0, 1.0, size=[batch_size, 100])
            x = self.generator.predict(noise)
            y = np.ones([batch_size, 1])
            self.discriminator.fit(x, y, nb_epoch=1, batch_size=32)
            y_hat = self.discriminator.predict(x)
            y_hat_idx = np.argmax(y_hat, axis=1)
            y_idx = np.argmax(y, axis=1)
            diff = y_idx - y_hat_idx
            n_tot = y.shape[0]
            n_rig = (diff == 0).sum()
            acc = n_rig * 100.0 / n_tot
            print("Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot))

        # Train both the discriminator and adversarial
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(0, 1.0, size=[16, 100])
        for j in range(epochs):
            for i in range(batch_num):
                images_train = self.x_train[np.random.randint(0,
                    self.x_train.shape[0], size=batch_size), :, :, :]
                noise = np.random.uniform(0, 1.0, size=[batch_size, 100])
                images_fake = self.generator.predict(noise)
                x = np.concatenate((images_train, images_fake))
                y = np.ones([2*batch_size, 1])
                y[batch_size:, :] = 0
                d_loss = self.discriminator.train_on_batch(x, y)

                y = np.ones([batch_size, 1])
                noise = np.random.uniform(0, 1.0, size=[batch_size, 100])
                a_loss = self.adversarial.train_on_batch(noise, y)
                log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
                log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                print(log_mesg)
                if save_interval > 0:
                    if (i+1) % save_interval == 0:
                        self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))
                        self.save_weights(self.discriminator, self.save_discr)
                        self.save_weights(self.adversarial, self.save_adver)
                        self.save_weights(self.generator, self.save_gener)


    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'out/cifar10/cifar.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(0, 1.0, size=[samples, 100])
            else:
                filename = "out/cifar10/set_{}_{}.png".format(self.category, step)
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        # create a grid of 3x3 images
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(toimage(images[i]))
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    def save_weights(self, model, filename):
        if not path.exists('weights'):
            os.mkdir('weights')
        # serialize model to JSON
        # model_json = model.to_json()
        # with open("{}.json".format(graph_name), "w") as json_file:
        #     json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("weights/{}_weights.h5".format(filename))
        print("Saved weights to disk")

    def load_weights(self, model, filename):
        if not path.exists('weights'):
            os.mkdir('weights')

        location = 'weights/{}_weights.h5'.format(filename)
        if path.exists(location):
            print('--- loading weights from "{}"'.format(location))
            model.load_weights(location)

        return model


if __name__ == '__main__':
    dcgan = CIFAR10_DCGAN(category=1)

    timer = ElapsedTimer()
    dcgan.train(epochs=100000, batch_size=256, save_interval=1000)
    timer.elapsed_time()
    dcgan.plot_images(fake=True)
    dcgan.plot_images(fake=False, save2file=True)
