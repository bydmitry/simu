#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:46:15 2017

@author: bychkov
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import backend as K

input_img = Input(shape=(40, 40, 3))

x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (5, 5, 32) i.e. 400-dimensional

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#----------------------------------------------------------------------
# Load mnist data:
#----------------------------------------------------------------------

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

#----------------------------------------------------------------------
# Load synthetic  data:
#----------------------------------------------------------------------
data = np.load('synthetic_imgs/test.npz')

train_x = data['train_x']
train_y = data['train_y']
val_x   = data['valid_x']
val_y   = data['valid_y']
test_x  = data['test_x']
test_y  = data['test_y']

train_x = np.transpose(train_x, (0,2,3,1))
val_x   = np.transpose(val_x, (0,2,3,1))
test_x  = np.transpose(test_x, (0,2,3,1))

w_size = 40
s_size = 35

N = train_x.shape[0]
patches_lst = list()
for p in range(train_x.shape[0]):
    tt = extract_patches(
            train_x[p,...],
            (w_size,w_size,3), (s_size,s_size,3) )
    patches_lst.append( np.squeeze(tt) )

patches_train = np.stack(patches_lst, axis=0)
patches_train = np.reshape(patches_train, (N * 3*3,w_size,w_size,3))

N = test_x.shape[0]
patches_lst = list()
for p in range(test_x.shape[0]):
    tt = extract_patches(
            test_x[p,...],
            (w_size,w_size,3), (s_size,s_size,3) )
    patches_lst.append( np.squeeze(tt) )

patches_test = np.stack(patches_lst, axis=0)
patches_test = np.reshape(patches_test, (N*3*3,w_size,w_size,3))

N = val_x.shape[0]
patches_lst = list()
for p in range(val_x.shape[0]):
    tt = extract_patches(
            val_x[p,...],
            (w_size,w_size,3), (s_size,s_size,3) )
    patches_lst.append( np.squeeze(tt) )

patches_val = np.stack(patches_lst, axis=0)
patches_val = np.reshape(patches_val, (N*3*3,w_size,w_size,3))


#----------------------------------------------------------------------
# Experiment settings:
#----------------------------------------------------------------------

span = 3000
autoencoder.fit(patches_train[:span], patches_train[:span],
                epochs=10,
                batch_size=64,
                shuffle=True,
                validation_data=(patches_test[:span], patches_test[:span]))


#----------------------------------------------------------------------
# Check reconstructions:
#----------------------------------------------------------------------
n = 10
decoded_test = autoencoder.predict(patches_test[:n])
decoded_train = autoencoder.predict(patches_train[:n])

# Test patches:
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(patches_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Train patches:
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(patches_train[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_train[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()