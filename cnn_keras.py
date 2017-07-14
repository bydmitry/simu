#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:38:34 2017

@author: bychkov
"""

import os
import theano
import shutil
import pickle
import platform

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
if platform.node() == 'lm4-fim-mg3qd.pc.helsinki.fi':
    plt.ioff()
plt.style.use('ggplot')

#import skimage
import pylab as pl
import numpy as np
#import pandas as pd

import sklearn
from sklearn.metrics import roc_auc_score

#from matplotlib import offsetbox
#from sklearn.feature_extraction import image

import keras
from keras import backend as K

os.environ['KERAS_BACKEND'] = 'theano'
if platform.node() == 'lm4-fim-mg3qd.pc.helsinki.fi':
    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32'

np.random.seed(1337) # for reproducibility
#np.set_printoptions(precision=5, suppress=True)

#==============================================================================
# Custom model initializer Class:
#==============================================================================
class Builder(object):
    def __init__(self, opts, train_data, test_data, train_ids, test_ids):
        self.opts  = opts
        self.test_data  = test_data
        self.train_data = train_data
        self.train_ids  = train_ids
        self.test_ids   = test_ids

        # Initialize model architecture:
        self.model = opts['zoo_model']

        # Compile model:
        self.model.name = opts['exp_name']
        self.model.compile(loss = opts['loss'], optimizer = opts['optimizer'],metrics=['accuracy'])

        # Prepare working directory:
        self.wrk_dir    = os.path.join(opts['exp_path'], opts['exp_name'])
        self.state_dir  = os.path.join(self.wrk_dir, 'state')

        if self.opts['continue'] and os.path.exists(self.state_dir):
            # Continue training existing model:
            self.routines = self.EpochRoutine(builder=self, history=True)

            curr_epoch = self.routines.history['epoch'][-1]
            curr_state = os.path.join(self.state_dir, 'epoch_%s.h5' % str(curr_epoch))
            self.model.load_weights( curr_state )
            print('\nRestoring the model from: %s' % self.wrk_dir)
            print('continue from epoch: %s ' % str(curr_epoch))
        else:
            # Initialize a new model:
            print('\ninitializing a new model...')
            if os.path.exists(self.wrk_dir):
                shutil.rmtree(self.wrk_dir)

            # Create a working directory:
            os.makedirs(self.wrk_dir)
            os.makedirs(self.state_dir)

            # Save model structure:
            with open(os.path.join(self.wrk_dir, 'model.yml'), "w") as text_file:
                text_file.write( self.model.to_yaml() )

            self.routines = self.EpochRoutine(builder=self, history=False)

        print('Model initialized.')

    # ------------------------------------------------------------------------------
    def train(self, epoches, batch, verbose=1):
        history = self.model.fit( x = self.train_data[0],
                         y = self.train_data[1],
                         batch_size       = batch,
                         nb_epoch         = epoches,
                         verbose          = verbose,
                         validation_split = self.opts['val_split'],
                         #validation_data  = (self.test_data[0], self.test_data[1]),
                         callbacks        = [ self.routines ]  )

        return history

    # ------------------------------------------------------------------------------
    # EpochRoutine Class:
    # ------------------------------------------------------------------------------
    class EpochRoutine(keras.callbacks.Callback):
        def __init__(self, builder, history=False, *args, **kwargs):
            super(builder.EpochRoutine, self).__init__(*args, **kwargs)

            self.opts = builder.opts
            self.builder = builder

            if not history:
                self.history = dict()
                self.history['epoch']       = []
                self.history['train_loss']  = []
                self.history['valid_loss']  = []
                self.history['test_loss']   = []
                self.history['train_acc']   = []
                self.history['valid_acc']   = []
                self.history['test_acc']    = []

                self.epoch_count = 0
            else:
                history_path = os.path.join(builder.wrk_dir, 'history.pkl')
                self.history = pickle.load(open(history_path, 'rb'))
                self.epoch_count = self.history['epoch'][-1]

        # ------------------------------------------------------------------------------
        def on_epoch_end(self, epoch, logs={}):
            epoch += 1
            self.epoch_count += 1
            # Available entities:
            # epoch, logs, self.model, self.params, self.opts, self.history

            # Get statistics on test set:
            test_eval = self.model.model.evaluate(
                self.builder.test_data[0],
                self.builder.test_data[1],
                batch_size=20
            )

            # Save training performance (loss):
            self.history['epoch'].append( self.epoch_count )
            self.history['train_loss'].append( logs.get('loss') )
            self.history['valid_loss'].append( logs.get('val_loss') )
            self.history['test_loss'].append( test_eval[0] )
            self.history['train_acc'].append( logs.get('acc') )
            self.history['valid_acc'].append( logs.get('val_acc') )
            self.history['test_acc'].append( test_eval[1] )

            # Visualize loss at each apoch:
            self.plot_loss(y1='train_loss', y2='valid_loss', y3='test_loss')
            self.plot_accuracy(y1='train_acc', y2='valid_acc', y3='test_acc')

            # Dump model weights & update training history:
            if ( self.epoch_count % self.opts['dump_freq'] == 0 or self.epoch_count == 1):
                self.dump_state()
                self.dump_history()
        # ------------------------------------------------------------------------------

        def dump_state(self):
            self.model.save_weights(
                os.path.join(self.builder.state_dir,'epoch_%d.h5' % self.epoch_count) )

        def dump_history(self):
            path = os.path.join(self.builder.wrk_dir, 'history.pkl')
            pickle.dump( self.history, open(path, 'wb') )

        def plot_accuracy(self, y1, y2, y3):
            file_name = 'Accuracy.png'
            # Make a plot:
            pl.figure()
            pl.title( 'Accuracy (1-error)' )
            pl.plot(self.history['epoch'], self.history[y1], label=y1, color='#348ABD')
            pl.plot(self.history['epoch'], self.history[y2], label=y2, color='#8EBA42')
            pl.plot(self.history['epoch'], self.history[y3], label=y3, color='#BA4252')
            pl.ylim(0.0, 1.0)
            pl.legend(loc='upper center', ncol=3)
            pl.savefig(os.path.join(self.builder.wrk_dir, file_name))
            plt.close()

        def plot_loss(self, y1, y2, y3):
            file_name = 'Loss.png'
            # Make a plot:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            lns1 = ax1.plot(self.history['epoch'], self.history[y1], label=y1, color='#348ABD')
            lns2 = ax2.plot(self.history['epoch'], self.history[y2], label=y2, color='#8EBA42')
            lns3 = ax2.plot(self.history['epoch'], self.history[y3], label=y3, color='#BA4252')
            ax1.set_ylabel(y1); ax2.set_ylabel('valid/test losses')
            # legend:
            lns = lns1+lns2+lns3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper center', ncol=3)
            plt.savefig(os.path.join(self.builder.wrk_dir, file_name))
            plt.close()

        def get_activations(self, x, model, layer, learn_phase = 0):
            func = K.function([model.layers[0].input, K.learning_phase()], [ model.layers[layer].output ])
            return func([x, learn_phase])

        def total_epoch_count(self):
            return self.epoch_count


#==============================================================================
# Model
#==============================================================================

import keras.models as models
from keras.layers import InputLayer
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2


model = models.Sequential()
model.add( InputLayer(input_shape=(3, 90, 90,), name='InputLayer') )

model.add( Convolution2D(32, 3, 3, activation='relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Convolution2D(32, 3, 3, activation='relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Convolution2D(64, 3, 3, activation='relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Convolution2D(64, 3, 3, activation='relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Flatten() )
model.add( Dense(256, activation='sigmoid') )
model.add( Dense(1, activation='sigmoid') )

#==============================================================================
# Main code:
#==============================================================================
# Dataset npz-file:
data = np.load('synthetic_imgs/first.npz')
class_names = np.array(['benign','cancer'])

train_x = data['train_x']
train_y = data['train_y']
test_x  = data['test_x']
test_y  = data['test_y']

#----------------------------------------------------------------------
# Set model & training parameters:
#----------------------------------------------------------------------
opts                    = dict()
opts['dump_freq']       = 1
opts['zoo_model']       = model
opts['val_split']       = 0.3
opts['loss']            = 'binary_crossentropy'
opts['optimizer']       = 'adadelta'
opts['exp_path']        = '/Users/bychkov/GDD/projects/simu'
opts['exp_name']        = 'first'
opts['continue']        = True

#----------------------------------------------------------------------
# Define the model:
#----------------------------------------------------------------------
network = Builder( opts = opts,
                    train_data = ( train_x, train_y ),
                    test_data  = ( test_x, test_y ),
                    train_ids  = data['train_id'],
                    test_ids   = data['test_id']   )

history = network.train(epoches=15, batch=25, verbose=1)

#==============================================================================
# Done.
#==============================================================================