from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from keras.layers import Add, ConvLSTM2D, Conv3D, UpSampling3D, Merge
from keras.layers import TimeDistributed

from keras.models import Model, Sequential
from keras import backend as K


n_convfilter = [96, 128, 256, 256, 256, 256]
n_fc_filters = [1024]
n_lstm_kernel = [4]
n_deconvfilter = [128, 128, 128, 64, 32, 2]

timesteps = 8
x = (timesteps, 512, 512, 3)  # using `channels_last` image data format

#...make this work?
model1  = Sequential()
model1.add(TimeDistributed(Conv2D(n_convfilter[0], (7, 7)), input_shape=x))
model1.add(TimeDistributed(LeakyReLU(alpha=.01)))
model1.add(TimeDistributed(Conv2D(n_convfilter[0], (3, 3))))
model1.add(TimeDistributed(LeakyReLU(alpha=.01)))
model1.add(TimeDistributed(MaxPooling2D((2, 2))))

model1.add(TimeDistributed(Conv2D(n_convfilter[1], (3, 3))))
model1.add(TimeDistributed(LeakyReLU(alpha=.01)))
model1.add(TimeDistributed(Conv2D(n_convfilter[1], (3, 3))))
model1.add(TimeDistributed(LeakyReLU(alpha=.01)))
model1.add(TimeDistributed(Conv2D(n_convfilter[1], (1, 1))))
model2 = Sequential()
model2.add(TimeDistributed(Conv2D(n_convfilter[1], (1, 1,))))


print(model2.output_shape)
# encoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# conv1a = Conv2D(n_convfilter[0], (7, 7), activation='linear', padding='same')(x)
# leak1a = LeakyReLU(alpha=.01)(conv1a)
# conv1b = Conv2D(n_convfilter[0], (3, 3), activation='linear', padding='same')(leak1a)
# leak1b = LeakyReLU(alpha=.01)(conv1b)
# pool1  = MaxPooling2D((2, 2), padding='same')(leak1b)
#
# conv2a = Conv2D(n_convfilter[1], (3, 3), activation='linear', padding='same')(pool1)
# leak2a = LeakyReLU(alpha=.01)(conv2a)
# conv2b = Conv2D(n_convfilter[1], (3, 3), activation='linear', padding='same')(leak2a)
# leak2b = LeakyReLU(alpha=.01)(conv2b)
# conv2c = Conv2D(n_convfilter[1], (1, 1), activation='linear', padding='same')(pool1)
# added2 = Add()([leak2b, conv2c])
# pool2  = MaxPooling2D((2, 2), padding='same')(added2)
#
# conv3a = Conv2D(n_convfilter[2], (3, 3), activation='linear', padding='same')(pool2)
# leak3a = LeakyReLU(alpha=.01)(conv3a)
# conv3b = Conv2D(n_convfilter[2], (3, 3), activation='linear', padding='same')(leak3a)
# leak3b = LeakyReLU(alpha=.01)(conv3b)
# conv3c = Conv2D(n_convfilter[2], (1, 1), activation='linear', padding='same')(pool2)
# added3 = Add()([leak3b, conv3c])
# pool3  = MaxPooling2D((2, 2), padding='same')(conv3c)
#
# conv4a = Conv2D(n_convfilter[3], (3, 3), activation='linear', padding='same')(pool3)
# leak4a = LeakyReLU(alpha=.01)(conv4a)
# conv4b = Conv2D(n_convfilter[3], (3, 3), activation='linear', padding='same')(leak4a)
# leak4b = LeakyReLU(alpha=.01)(conv4b)
# pool4  = MaxPooling2D((2, 2), padding='same')(leak4b)
#
# conv5a = Conv2D(n_convfilter[4], (3, 3), activation='linear', padding='same')(pool4)
# leak5a = LeakyReLU(alpha=.01)(conv5a)
# conv5b = Conv2D(n_convfilter[4], (3, 3), activation='linear', padding='same')(leak5a)
# leak5b = LeakyReLU(alpha=.01)(conv5b)
# conv5c = Conv2D(n_convfilter[4], (1, 1), activation='linear', padding='same')(conv5b)
# added5 = Add()([leak5b, conv5c])
# pool5  = MaxPooling2D((2, 2), padding='same')(added5)
#
# conv6a = Conv2D(n_convfilter[5], (3, 3), activation='linear', padding='same')(pool5)
# leak6a = LeakyReLU(alpha=.01)(conv6a)
# conv6b = Conv2D(n_convfilter[5], (3, 3), activation='linear', padding='same')(conv6a)
# leak6b = LeakyReLU(alpha=.01)(conv6b)
# pool6  = MaxPooling2D((2, 2), padding='same')(leak6b)
#
# # flat6 = Flatten()(pool6)
# # dense6 = Dense(n_fc_filters[0])(flat6)
# # merge6 = Merge([pool6, pool6], mode='dot', dot_axes=2)
# # timed7 = TimeDistributed(dense6)
#
# encoded= ConvLSTM2D(n_deconvfilter[0], (3, 3), return_sequences=True)(pool6)
#
# upsmp7 = UpSampling3D((3, 3, 3), padding='same')(encoded)
# conv7a = Conv3D(n_deconvfilter[1], (3, 3))(upsmp7)
# leak7a = LeakyReLU(alpha=.01)(conv7a)
# conv7b = Conv3D(n_deconvfilter[1], (3, 3))(leak7a)
# leak7b = LeakyReLU(alpha=.01)(conv7b)
# added7 = Add()([leak7b, upsmp7])

## ORIGINAL
# conv1a = Conv2D(n_convfilter[0], (7, 7), activation='linear', padding='same')(x)
# leak1a = LeakyReLU(alpha=.01)(conv1a)
# conv1b = Conv2D(n_convfilter[0], (3, 3), activation='linear', padding='same')(leak1a)
# leak1b = LeakyReLU(alpha=.01)(conv1b)
# pool1  = MaxPooling2D((2, 2), padding='same')(leak1b)
#
# conv2a = Conv2D(n_convfilter[1], (3, 3), activation='linear', padding='same')(pool1)
# leak2a = LeakyReLU(alpha=.01)(conv2a)
# conv2b = Conv2D(n_convfilter[1], (3, 3), activation='linear', padding='same')(leak2a)
# leak2b = LeakyReLU(alpha=.01)(conv2b)
# conv2c = Conv2D(n_convfilter[1], (1, 1), activation='linear', padding='same')(pool1)
# added2 = Add()([leak2b, conv2c])
# pool2  = MaxPooling2D((2, 2), padding='same')(added2)
#
# conv3a = Conv2D(n_convfilter[2], (3, 3), activation='linear', padding='same')(pool2)
# leak3a = LeakyReLU(alpha=.01)(conv3a)
# conv3b = Conv2D(n_convfilter[2], (3, 3), activation='linear', padding='same')(leak3a)
# leak3b = LeakyReLU(alpha=.01)(conv3b)
# conv3c = Conv2D(n_convfilter[2], (1, 1), activation='linear', padding='same')(pool2)
# added3 = Add()([leak3b, conv3c])
# pool3  = MaxPooling2D((2, 2), padding='same')(conv3c)
#
# conv4a = Conv2D(n_convfilter[3], (3, 3), activation='linear', padding='same')(pool3)
# leak4a = LeakyReLU(alpha=.01)(conv4a)
# conv4b = Conv2D(n_convfilter[3], (3, 3), activation='linear', padding='same')(leak4a)
# leak4b = LeakyReLU(alpha=.01)(conv4b)
# pool4  = MaxPooling2D((2, 2), padding='same')(leak4b)
#
# conv5a = Conv2D(n_convfilter[4], (3, 3), activation='linear', padding='same')(pool4)
# leak5a = LeakyReLU(alpha=.01)(conv5a)
# conv5b = Conv2D(n_convfilter[4], (3, 3), activation='linear', padding='same')(leak5a)
# leak5b = LeakyReLU(alpha=.01)(conv5b)
# conv5c = Conv2D(n_convfilter[4], (1, 1), activation='linear', padding='same')(conv5b)
# added5 = Add()([leak5b, conv5c])
# pool5  = MaxPooling2D((2, 2), padding='same')(added5)
#
# conv6a = Conv2D(n_convfilter[5], (3, 3), activation='linear', padding='same')(pool5)
# leak6a = LeakyReLU(alpha=.01)(conv6a)
# conv6b = Conv2D(n_convfilter[5], (3, 3), activation='linear', padding='same')(conv6a)
# leak6b = LeakyReLU(alpha=.01)(conv6b)
# pool6  = MaxPooling2D((2, 2), padding='same')(leak6b)
#
# # flat6 = Flatten()(pool6)
# # dense6 = Dense(n_fc_filters[0])(flat6)
# # merge6 = Merge([pool6, pool6], mode='dot', dot_axes=2)
# # timed7 = TimeDistributed(dense6)
#
# encoded= ConvLSTM2D(n_deconvfilter[0], (3, 3), return_sequences=True)(pool6)
#
# upsmp7 = UpSampling3D((3, 3, 3), padding='same')(encoded)
# conv7a = Conv3D(n_deconvfilter[1], (3, 3))(upsmp7)
# leak7a = LeakyReLU(alpha=.01)(conv7a)
# conv7b = Conv3D(n_deconvfilter[1], (3, 3))(leak7a)
# leak7b = LeakyReLU(alpha=.01)(conv7b)
# added7 = Add()([leak7b, upsmp7])




# x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)
#
# # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)