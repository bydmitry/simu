#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:33:08 2017

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
cnn.add(Reshape((36,D)))

# --- Attentive LSTM ---
X_t            = Input((K2,D), name='X_t')
readout_input  = Input((attentive_lstm_dim,), name='readout_input')

h_tm1          = Input((attentive_lstm_dim,), name='h_tm1')
c_tm1          = Input((attentive_lstm_dim,), name='c_tm1')

x_kernel  = Conv1D(filters=1, kernel_size=1, strides=1)
h_kernel  = Dense(K2, activation='linear')

W_xt           = x_kernel(X_t)
W_xt           = Reshape((K2,))(W_xt)
W_ht           = h_kernel(readout_input)

att_scores     = add([W_xt,W_ht])
att_mask       = Activation(K.softmax, name='att_mask')(att_scores)

lstms_input    = dot([att_mask, X_t], axes=(1,1))

cells = [ LSTMCell(attentive_lstm_dim) for _ in range(attentive_lstm_depth) ]

lstms_output, h, c = lstms_input, h_tm1, c_tm1
for cell in cells:
    lstms_output, h, c = cell([lstms_output, h, c])

attentive_lstm = RecurrentModel(input = X_t, output = lstms_output,
                     initial_states    = [h_tm1, c_tm1],  
                     final_states      = [h, c], 
                     readout_input     = readout_input,
                     return_states     = False,
                     return_sequences  = True)

#--- Full Model ---#
fmap_seq   = TimeDistributed(cnn)(img_seq)
lstm_out1  = attentive_lstm(fmap_seq)
lstm_out2  = LSTM(2,  activation='linear')(lstm_out1)
hazard     = Dense(1, activation='linear')(lstm_out2)

model = Model(img_seq, hazard)
#model.compile(loss='mean_squared_error', optimizer='adam')

#model.fit(patches, target, epochs=45, batch_size=50)

 #--- Test ---#
preds = model.predict( np.random.random((5,9,40,40,3)) )
preds.shape

#--- Misc ---#
cnn_seq  = K.function(
            inputs  = [ model.layers[0].input ], 
            outputs = [ model.layers[1].output ] )

att_seq  = K.function(
            inputs  = [ model.layers[0].input ], 
            outputs = [ model.layers[5].output ] )

get_mask = K.function(
        inputs  = [ model.layers[5].model.layers[0].input,
                    model.layers[5].model.layers[2].input ], 
        outputs = [ model.layers[5].model.layers[6].output ] )


fake_input  = patches[0:1,...]
fmaps  = cnn_seq( [ fake_input ] )[0]
hhs    = att_seq( [ fake_input ] )[0]

im_idx   = 0
mask_lst = list()
for tile in range(1,9):
    mask_lst.append(
        np.reshape( 
            get_mask([ fmaps [ im_idx, tile:tile+1, ... ] , 
                       hhs [ im_idx, tile-1:tile, ... ]     ])[0],
            (6,6))
    )
    
masks = np.stack(mask_lst, axis=0)
#----------------------------------------------------------------------
# Experiment settings:
#----------------------------------------------------------------------

opts                    = dict()
opts['dump_freq']       = 1
opts['plot_freq']       = 1
opts['krs_model']       = model
opts['loss']            = partial_likelihood
opts['metrics_l']       = None
opts['optimizer']       = 'adadelta'
opts['exp_path']        = '/Users/bychkov/GDD/projects/simu/models'
opts['exp_name']        = 'att_1'
opts['continue']        = True


network = Builder(  opts = opts,
                    train_data = ( patches_train, train_y ),
                    test_data  = ( patches_test, test_y ),
                    val_data   = ( patches_val, val_y )  ) 

#----------------------------------------------------------------------
# Train:
#----------------------------------------------------------------------

history = network.train(epoches=15, batch=1000, shuffle=True, verbose=1)

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