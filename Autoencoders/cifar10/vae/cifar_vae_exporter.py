import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import cifar10

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


batch_size = 100
original_dim = 3072
latent_dim = 2
intermediate_dim = 1024
epochs = 100
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = load_weights(Model(x, x_decoded_mean), 'cifar_vae')
vae.summary()

for i in vae.layers:
    if vae.get_layer(name=i.name).get_weights():
        print(i.name)
        weights = vae.get_layer(name=i.name).get_weights()[0]
        bias = vae.get_layer(name=i.name).get_weights()[1]
        np.savetxt("export/{}_weights.csv".format(i.name), weights, delimiter=",")
        np.savetxt("export/{}_bias.csv".format(i.name), bias, delimiter=",")