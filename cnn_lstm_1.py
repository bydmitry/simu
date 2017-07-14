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


#----------------------------------------------------------------------
# Load data:
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

#----------------------------------------------------------------------
# Prepare data:
#----------------------------------------------------------------------
w_size = 40
s_size = 35

N = train_x.shape[0]
patches_lst = list()
for p in range(train_x.shape[0]):
    tt = extract_patches(
            train_x[1,...], 
            (w_size,w_size,3), (s_size,s_size,3) )
    patches_lst.append( np.squeeze(tt) )

patches_train = np.stack(patches_lst, axis=0)
patches_train = np.reshape(patches_train, (N,3*3,w_size,w_size,3))

N = test_x.shape[0]
patches_lst = list()
for p in range(test_x.shape[0]):
    tt = extract_patches(
            test_x[1,...], 
            (w_size,w_size,3), (s_size,s_size,3) )
    patches_lst.append( np.squeeze(tt) )

patches_test = np.stack(patches_lst, axis=0)
patches_test = np.reshape(patches_test, (N,3*3,w_size,w_size,3))

N = val_x.shape[0]
patches_lst = list()
for p in range(val_x.shape[0]):
    tt = extract_patches(
            val_x[1,...], 
            (w_size,w_size,3), (s_size,s_size,3) )
    patches_lst.append( np.squeeze(tt) )

patches_val = np.stack(patches_lst, axis=0)
patches_val = np.reshape(patches_val, (N,3*3,w_size,w_size,3))

#----------------------------------------------------------------------
# Keras model:
#----------------------------------------------------------------------

from keras.models import Model, Sequential
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

# --- CNN ---
img_seq = Input(shape=(None,40,40,3), name='img_seq')

cnn = Sequential()
cnn.add(Conv2D(16, (3, 3), activation='relu', input_shape=(40, 40, 3)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(16, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(D, (3, 3), activation='relu'))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(Conv2D(128, (4, 4), activation='relu'))
cnn.add(Flatten())

#cnn.add(Reshape((36,D)))


#--- Full Model ---#
fmap_seq   = TimeDistributed(cnn)(img_seq)
lstm_out1  = LSTM(512,  activation='softsign', return_sequences=False)(fmap_seq)
#lstm_out2  = LSTM(128,  activation='softsign')(lstm_out1)
hazard     = Dense(1, activation='linear')(lstm_1)

model = Model(img_seq, hazard)
model.compile(loss='mean_squared_error', optimizer='adadelta')

model.fit(patches_train[:250], train_y[:250,0], epochs=777, batch_size=50)

 #--- Vis ---#
preds = np.squeeze(model.predict( patches_train[:250] ))
plt.scatter(train_y[:250,0], preds)
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