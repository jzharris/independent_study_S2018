import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense
from keras.models import Model

import os
import os.path as path
def save_weights(model, filename):
    if not path.exists('weights'):
        os.mkdir('weights')
    model.save_weights("weights/{}_weights.h5".format(filename))
    print('--- weights saved to "weights/{}_weights.h5"'.format(filename))


def load_weights(model, filename):
    if not path.exists('weights'):
        os.mkdir('weights')

    location = 'weights/{}_weights.h5'.format(filename)
    if path.exists(location):
        print('--- loading weights from "{}"'.format(location))
        model.load_weights(location)

    return model

# from scipy.misc import toimage
# def import_cifar10(category=0, plot=False):
#     filter_train = np.load('../datasets/cifar10/set_{}.npy'.format(category))
#
#     if(plot):
#         images = filter_train[:16]
#         # create a grid of 3x3 images
#         plt.figure(figsize=(10, 10))
#         for i in range(images.shape[0]):
#             plt.subplot(4, 4, i + 1)
#             plt.imshow(toimage(images[i]))
#             plt.axis('off')
#         plt.tight_layout()
#         plt.show()
#
#     return filter_train

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

for i in autoencoder.layers:
    if autoencoder.get_layer(name=i.name).get_weights():
        print(i.name)
        weights = autoencoder.get_layer(name=i.name).get_weights()[0]
        bias = autoencoder.get_layer(name=i.name).get_weights()[1]
        np.savetxt("export/e_0/{}_weights.csv".format(i.name), weights, delimiter=",")
        np.savetxt("export/e_0/{}_bias.csv".format(i.name), bias, delimiter=",")


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print x_train.shape
# print x_test.shape

autoencoder.fit(x_train, x_train,
                epochs=25,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

for i in autoencoder.layers:
    if autoencoder.get_layer(name=i.name).get_weights():
        print(i.name)
        weights = autoencoder.get_layer(name=i.name).get_weights()[0]
        bias = autoencoder.get_layer(name=i.name).get_weights()[1]
        np.savetxt("export/e_25/{}_weights.csv".format(i.name), weights, delimiter=",")
        np.savetxt("export/e_25/{}_bias.csv".format(i.name), bias, delimiter=",")


autoencoder.fit(x_train, x_train,
                epochs=25,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

for i in autoencoder.layers:
    if autoencoder.get_layer(name=i.name).get_weights():
        print(i.name)
        weights = autoencoder.get_layer(name=i.name).get_weights()[0]
        bias = autoencoder.get_layer(name=i.name).get_weights()[1]
        np.savetxt("export/e_50/{}_weights.csv".format(i.name), weights, delimiter=",")
        np.savetxt("export/e_50/{}_bias.csv".format(i.name), bias, delimiter=",")


# # encode and decode some digits
# # note that we take them from the *test* set
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
#
# # use Matplotlib (don't ask)
# import matplotlib.pyplot as plt
#
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
#
