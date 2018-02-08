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

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop', loss='binary_crossentropy')
vae.summary()


# train the VAE on CIFAR images
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_data = import_cifar10(1)
# msk = np.random.rand(x_data.shape[0]) < 0.8
# x_train = x_data[msk]
# x_test = x_data[~msk]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_test = y_test.reshape((len(y_test)))

########################################################################
from keras.callbacks import TensorBoard

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir='logs/cifar_vae')])

# save weights!
save_weights(vae, 'cifar_vae')
########################################################################

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test) # start here
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
img_size = 32
figure = np.zeros((img_size * n, img_size * n, 3))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        cifar = x_decoded.reshape(img_size, img_size, 3)
        figure[i * img_size: (i + 1) * img_size,
               j * img_size: (j + 1) * img_size] = cifar

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()