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

from builder import *

import att_utils

#----------------------------------------------------------------------
# Load data:
#----------------------------------------------------------------------
step_size = 35
tile_size = 40

data_dict = att_utils.load_synthetic_imgs('synthetic_imgs/test.npz')
data_dict = att_utils.extract_tiles(data_dict, tile_size, step_size)

#----------------------------------------------------------------------
# Keras model:
#----------------------------------------------------------------------

import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
from keras.layers.merge import multiply, add, concatenate, dot
from keras.regularizers import l1, l2
from keras import backend as K

from recurrentshop.cells import LSTMCell
from recurrentshop import RecurrentModel
from keras.layers.convolutional import Conv1D

# --- Main input --
input_patche_seq = Input(shape=(None,tile_size,tile_size,3), name='input_patche_seq')

# --- Inception ---
input_img = Input(shape=(tile_size, tile_size, 3), name='input_img')

tow_1 = Conv2D(8, (1,1), padding='same', activation='relu', name='t1_c1')(input_img)
tow_1 = Conv2D(8, (3,3), padding='same', activation='relu', name='t1_c2')(tow_1)

tow_2 = Conv2D(8, (1,1), padding='same', activation='relu', name='t2_c1')(input_img)
tow_2 = Conv2D(8, (5,5), padding='same', activation='relu', name='t2_c2')(tow_2)

tow_3 = MaxPooling2D((3,3), strides=(1,1), padding='same', name='t3_mp')(input_img)
tow_3 = Conv2D(8, (1,1), padding='same', activation='relu', name='t3_c1')(tow_3)

incpt = keras.layers.concatenate([tow_1, tow_2, tow_3], axis=3, name='incpt')

incpt = Conv2D(32, (3,3), padding='same', activation='relu')(incpt)
incpt = MaxPooling2D((2,2))(incpt)
incpt = Reshape((400,32))(incpt)

inception = Model(input_img, incpt)
inception.summary()

# --- CNN ---
cnn = Sequential()
cnn.add(Conv2D(8, (3, 3), activation='relu', input_shape=(tile_size, tile_size, 3)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(16, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(D, (3, 3), activation='relu'))
cnn.add(Reshape((36,D)))

# --- Attentive LSTM ---
K2 = 20*20  # num of locations (within one tile/time-step)
D  = 32  # num of feature maps

attentive_lstm_dim   = 48
attentive_lstm_depth = 2

X_t            = Input((K2,D), name='X_t')
readout_input  = Input((attentive_lstm_dim,), name='readout_input')

h_tm1          = Input((attentive_lstm_dim,), name='h_tm1')
c_tm1          = Input((attentive_lstm_dim,), name='c_tm1')

x_kernel  = Conv1D(filters=1, kernel_size=1, strides=1, activity_regularizer=l1(0.1))
h_kernel  = Dense(K2, activation='linear', activity_regularizer=l1(0.1))

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
fmap_seq   = TimeDistributed(inception)(input_patche_seq)
lstm_out1  = attentive_lstm(fmap_seq)
lstm_out2  = LSTM(8,  activation='tanh')(lstm_out1)
hazard     = Dense(1, activation='linear')(lstm_out2)

model = Model(input_patche_seq, hazard)
model.compile(loss='mean_squared_error', optimizer='adadelta')
#model.compile(loss=partial_likelihood, optimizer='adadelta')
#----------------------------------------------------------------------
# Fit:
#----------------------------------------------------------------------
nn = [7, 9, 31, 24] #300

model.fit(
        x           = data_dict['train_x_p'][:350,],
        y           = data_dict['train_y'][:350,0],
        #validation_data = (
        #        data_dict['valid_x_p'][:550,],
        #        data_dict['valid_y'][:550,0]    ),
        validation_split = 0.3,
        epochs      = 3,
        batch_size  = 10 )

#--- Vis ---#
preds = np.squeeze(model.predict( data_dict['train_x_p'][:150,] ))
plt.scatter( data_dict['train_y'][:150, 0], preds )

preds = np.squeeze(model.predict( data_dict['test_x_p'][:150,] ))
plt.scatter( data_dict['test_y'][:150, 0], preds )


#----------------------------------------------------------------------
# Overlay masks with images:
#----------------------------------------------------------------------
cnn_seq  = K.function(
            inputs  = [ model.layers[0].input ],
            outputs = [ model.layers[1].output ] )

att_seq  = K.function(
            inputs  = [ model.layers[0].input ],
            outputs = [ model.layers[5].get_output_at(0) ] )
# model.layers[5].output

get_mask = K.function(
        inputs  = [ model.layers[5].model.layers[0].input,
                    model.layers[5].model.layers[2].input ],
        outputs = [ model.layers[5].model.layers[6].output ] )

#------------------------------------------------------------

idx = np.random.randint(low=0, high = 350, size=4)

vis_img     = data_dict['train_x'][idx]

check_imgs  = data_dict['train_x_p'][idx]
fmaps       = cnn_seq( [ check_imgs ] )[0]
hhs         = att_seq( [ check_imgs ] )[0]

print check_imgs.shape
print fmaps.shape
print hhs.shape

num_imgs  = check_imgs.shape[0]
num_tiles = check_imgs.shape[1]
tile_h    = check_imgs.shape[2]
tile_w    = check_imgs.shape[3]
loc_grid  = int( np.sqrt(K2) )

#------------------------------------------------------------
# Iterate over Images:
mask_images = list()
for ind in range(num_imgs):
    tile_lst = list()
    # First tile mask is not available, so just zeros()
    tile_lst.append( np.zeros((loc_grid, loc_grid), dtype='float32') )
    # Iterate over N-1 Tiles:
    for tile in range(1,num_tiles):
        tile_lst.append(
            np.reshape(
                get_mask([ fmaps [ ind, tile:tile+1, ... ] ,
                           hhs [ ind, tile-1:tile, ... ]     ])[0],
                (loc_grid, loc_grid), 'C')
        )

    tiles = np.stack(tile_lst, axis=0)

    upscaled_tiles = att_utils.upscale_mask_tiles(tiles, (tile_h, tile_w))
    mask_images.append(
        att_utils.reconstruct_from_tiles_2d(upscaled_tiles, step_size, (111,111))
    )
masks = np.stack( mask_images, axis=0 )
#------------------------------------------------------------

print vis_img.shape
print masks.shape

# Visualize 2 x 2 Grid:
fig = plt.figure(frameon=False)
for im_ind in range(4):
    sub_i = 220 + im_ind + 1
    plt.subplot(sub_i)
    plt.imshow(
            att_utils.rgb2gray(vis_img[im_ind]),
            cmap=plt.cm.gray, alpha=0.35 )
    plt.imshow(masks[im_ind], alpha=.7)
    #plt.imshow(masks[im_ind], alpha=.7, vmin=0, vmax=255)
    print np.max( masks[im_ind] )
    plt.colorbar()

# Plot single image:
im_ind = 0
fig = plt.figure(frameon=False)
plt.imshow( att_utils.rgb2gray(vis_img[im_ind]),
            cmap=plt.cm.gray, alpha=0.5 )
plt.imshow(masks[im_ind], alpha=.6)
print np.max( masks[im_ind] )
plt.colorbar()


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
                    val_data   = ( patches_val, val_y )   )

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