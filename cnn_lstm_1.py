#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:40:42 2017

@author: bychkov
"""


import plotly
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.image import extract_patches
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from builder import *

import att_utils


#----------------------------------------------------------------------
# Load data:
#----------------------------------------------------------------------
data_dict = att_utils.load_synthetic_imgs('synthetic_imgs/test.npz')
data_dict = att_utils.extract_tiles(data_dict, 40, 35)

#----------------------------------------------------------------------
# Keras model:
#----------------------------------------------------------------------

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Lambda
from keras.layers.merge import multiply, add, concatenate, dot
from keras import backend as K

from recurrentshop.cells import LSTMCell
from recurrentshop import RecurrentModel
from keras.layers.convolutional import Conv1D

# --- Some hyper-parameters ---

K2 = 36 # num of locations (within one tile/time-step)
D  = 32  # num of feature maps

attentive_lstm_dim   = 16
attentive_lstm_depth = 2

#----------------------------------------------------------------------
# CNN from scratch:
#----------------------------------------------------------------------

img_seq = Input(shape=(None,40,40,3), name='img_seq')

cnn = Sequential()
cnn.add(Conv2D(8, (3, 3), activation='relu', input_shape=(40, 40, 3)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(8, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(16, (3, 3), activation='relu'))
cnn.add(Conv2D(16, (3, 3), activation='relu'))
cnn.add(Conv2D(32, (4, 4), activation='relu'))
cnn.add(Flatten())

#cnn.add(Reshape((36,D)))

#----------------------------------------------------------------------
# CNN pretrained as autoencoder:
#----------------------------------------------------------------------
autoencoder = load_model('models/conv_autoencoder/cae_model.h5')

img_seq = Input(shape=(None,40,40,3), name='img_seq')

code = autoencoder.get_layer(index=6).output
code = Flatten()(code)
cnn = Model(inputs=autoencoder.input, outputs=code)

bottleneck = cnn.predict(patches_train[0,(1,3,5),])

#----------------------------------------------------------------------
# Full model with LSTM:
#----------------------------------------------------------------------
fmap_seq   = TimeDistributed(cnn)(img_seq)
lstm_out1  = LSTM(64, activation='softsign', return_sequences=True)(fmap_seq)
lstm_out2  = LSTM(32, activation='softsign', return_sequences=False)(lstm_out1)
hazard     = Dense(1, activation='linear')(lstm_out2)

model = Model(img_seq, hazard)
model.compile(loss='mean_squared_error', optimizer='adadelta')

nn = 150
model.fit(patches_train[:nn,], train_y[:nn,0], epochs=3, batch_size=10)

 #--- Vis ---#
preds = np.squeeze(model.predict( patches_train[:nn,] ))
plt.scatter(train_y[:nn,0], preds)

#--- Test ---#
preds = model.predict( np.random.random((5,9,40,40,3)) )
preds.shape

#----------------------------------------------------------------------
# Experiment settings:
#----------------------------------------------------------------------

opts                    = dict()
opts['dump_freq']       = 1
opts['plot_freq']       = 1
opts['krs_model']       = model
opts['loss']            = partial_likelihood  # my_mse
opts['metrics_l']       = None
opts['optimizer']       = 'adadelta'
opts['exp_path']        = '/Users/bychkov/GDD/projects/simu/models'
opts['exp_name']        = 'cnn_lstm_1'
opts['continue']        = False


network = Builder(  opts = opts,
                    train_data = ( patches_train, train_y ),
                    test_data  = ( patches_test, test_y ),
                    val_data   = ( patches_val, val_y )  )

#----------------------------------------------------------------------
# Train:
#----------------------------------------------------------------------

history = network.train(epoches=3, batch=500, shuffle=True, verbose=1)

#----------------------------------------------------------------------
# Predict:
#----------------------------------------------------------------------

n_samples = 1
n_frames  = 9

frame_sequence = np.random.random((n_samples, n_frames, 40,40,3))

network.model.predict(frame_sequence)


preds = network.model.predict(patches_test)
preds[:10]
#----------------------------------------------------------------------
# Done.
#----------------------------------------------------------------------