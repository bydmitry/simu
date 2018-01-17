#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:16:58 2017

@author: bychkov
"""

import os
import theano
import theano.tensor as T
import tensorflow as tf
import shutil
import pickle
import platform
from collections import OrderedDict


import plotly
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go

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
from lifelines.utils import concordance_index
from scipy.stats import kendalltau, pearsonr
#from matplotlib import offsetbox
#from sklearn.feature_extraction import image

import keras
from keras import backend as K

#os.environ['KERAS_BACKEND'] = 'theano'
#if platform.node() == 'lm4-fim-mg3qd.pc.helsinki.fi':
#    os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32'

np.random.seed(1337) # for reproducibility
#np.set_printoptions(precision=5, suppress=True)

#==============================================================================
# Custom model initializer Class:
#==============================================================================
class Builder(object):
    def __init__(self, opts, train_data, test_data, val_data):
        self.opts  = opts

        # Reassign input data:
        self.data = dict()
        self.data['train'] = train_data
        self.data['valid'] = val_data
        self.data['test']  = test_data

        # Initialize model architecture:
        self.model = opts['krs_model']

        # Compile model:
        self.model.name = opts['exp_name']
        self.model.compile(
                loss       = opts['loss'],
                metrics    = opts['metrics_l'],
                optimizer  = opts['optimizer']   )

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

            # Save split ids:
            #self.save_ids()

            self.routines = self.EpochRoutine(builder=self, history=False)

        print('Model initialized.')

    # ------------------------------------------------------------------------------
    def train(self, epoches, batch, shuffle=True, verbose=1):
        history = self.model.fit(
                x   = self.data['train'][0],
                y   = self.data['train'][1],
                validation_data   = (
                        self.data['valid'][0],
                        self.data['valid'][1] ),
                validation_split  = 0.0,
                batch_size        = batch,
                epochs            = epoches,
                shuffle           = shuffle,
                verbose           = verbose,
                callbacks         = [ self.routines ] )

        return history
    # ------------------------------------------------------------------------------
    def save_ids(self):
        fname = os.path.join(self.wrk_dir, 'splits.csv')
        dfs   = list()
        for split in ['train', 'valid', 'test']:
            d  = OrderedDict()
            d['id']    = self.data[split][1][:,3]
            d['split'] = np.array([split] * d['id'].shape[0])
            dfs.append( pd.DataFrame( data=d ) )

        df = pd.concat(dfs, ignore_index=True, axis=0)
        df.sort_values(by='id', inplace=True)
        df.to_csv(fname, index = False)
    # ------------------------------------------------------------------------------
    # EpochRoutine Class:
    # ------------------------------------------------------------------------------
    class EpochRoutine(keras.callbacks.Callback):
        def __init__(self, builder, history=False, *args, **kwargs):
            super(builder.EpochRoutine, self).__init__(*args, **kwargs)

            self.opts     = builder.opts
            self.builder  = builder

            # This is to temporarily store predictions:
            self.preds    = dict()

            if not history:
                self.history = dict()
                self.history['epoch'] = []
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

            # Get loss on test set:
            test_loss = self.model.evaluate(
                    self.builder.data['test'][0],
                    self.builder.data['test'][1],
                    batch_size = self.builder.data['test'][0].shape[0],
                    verbose = 0 )

            # Save training-step performance:
            self.history['epoch'].append( self.epoch_count )
            self.history.setdefault('train_loss',[]).append( logs.get('loss') )
            self.history.setdefault('valid_loss',[]).append( logs.get('val_loss') )
            self.history.setdefault('test_loss',[]).append( test_loss )

            # Get accuracy statistics on train/val/test:
            self.custom_eval()

            # Visualize loss at each apoch:
            if self.epoch_count % self.opts['plot_freq'] == 0:
                self.plot_history('loss')
                self.plot_history('cindex', ylim=[0,1])
                self.plot_history('kentau', ylim=[-1,1])
                self.plot_history('pearsonr', ylim=[-1,1])
                self.plot_scatter()

            # Dump model weights & update training history:
            if ( self.epoch_count % self.opts['dump_freq'] == 0 or self.epoch_count == 1):
                self.dump_state()
                self.dump_history()
        # ------------------------------------------------------------------------------
        def custom_eval(self):
            self.preds = dict()
            # Get ptrdictions for all splits:
            for split in ['train', 'valid', 'test']:
                self.preds[split] = np.squeeze( self.model.predict(
                        self.builder.data[split][0],
                        batch_size = self.builder.data[split][0].shape[0],
                        verbose    = 0 ) )

            suff = '_cindex'
            for split in ['train', 'valid', 'test']:
                self.history.setdefault(split+suff,[]).append(
                    concordance_index(
                        self.builder.data[split][1][:,1],   # t
                        -self.preds[split],                 # h_hat
                        self.builder.data[split][1][:,2]  ) # e
                )

            suff = '_kentau'
            for split in ['train', 'valid', 'test']:
                tau, p_value = kendalltau(
                        self.builder.data[split][1][:,0],
                        self.preds[split] )
                self.history.setdefault(split+suff,[]).append( tau )

            suff = '_pearsonr'
            for split in ['train', 'valid', 'test']:
                rcoeff = pearsonr(
                        self.builder.data[split][1][:,0],
                        self.preds[split] )
                self.history.setdefault(split+suff,[]).append( rcoeff[0] )


            return True
        # ------------------------------------------------------------------------------
        def dump_state(self):
            self.model.save_weights(
                os.path.join(self.builder.state_dir,'epoch_%d.h5' % self.epoch_count) )

        def dump_history(self):
            path = os.path.join(self.builder.wrk_dir, 'history.pkl')
            pickle.dump( self.history, open(path, 'wb') )
        # ------------------------------------------------------------------------------
        def plot_scatter(self):
            fname = os.path.join(self.builder.wrk_dir, 'scatter_plot.html')

            test = go.Scattergl(
                name    = 'Test set',
                x       = self.builder.data['test'][1][:,0],
                y       = self.preds['test'],
                mode    = 'markers',
                marker  = dict(size = 3)
            )
            train = go.Scattergl(
                name    = 'Train set',
                x       = self.builder.data['train'][1][:,0],
                y       = self.preds['train'],
                mode    = 'markers',
                marker  = dict(size = 3)
            )

            layout = go.Layout(
                title  = 'Predicted VS Real Hazard',
                xaxis  = dict( title='H', range = [-4.5,4.5]),
                yaxis  = dict( title='H_hat', range = [-4.5,4.5]) )

            fig = go.Figure( data=[train, test], layout=layout )
            py.plot(fig, filename=fname, auto_open=False)

        def plot_history(self, metrics='loss', ylim=None):
            fname = os.path.join(self.builder.wrk_dir, '%s.html' % metrics)

            # Colors:
            cl1 = '#348ABD'; cl2 = '#8EBA42'; cl3='#BA4252';
            pmode = 'lines+markers'

            XS = np.array(self.history['epoch'])
            train_loss = go.Scatter( name='Train %s' % metrics,
                yaxis='y1',
                x = XS, y = np.array(self.history['train_%s' % metrics]),
                line = dict(color = cl1), mode = pmode )

            valid_loss = go.Scatter(name='Valid %s' % metrics,
                yaxis='y2',
                x = XS, y = np.array(self.history['valid_%s' % metrics]),
                line = dict(color = cl2), mode = pmode )

            test_loss  = go.Scatter(name='Test %s' % metrics,
                yaxis='y3',
                x = XS, y = np.array(self.history['test_%s' % metrics]),
                line = dict(color = cl3), mode = pmode )

            data = [train_loss, valid_loss, test_loss]

            layout = go.Layout(
                title  = 'Training dynamics',
                yaxis  = dict( title='Train %s' % metrics,
                    titlefont = dict(color=cl1),
                    tickfont  = dict(color=cl1),
                    range     = ylim),
                yaxis2 = dict( title='Valid %s' % metrics,
                    titlefont = dict(color=cl2),
                    tickfont  = dict(color=cl2),
                    range     = ylim,
                    overlaying='y', side='right', position=0.9, anchor ='free'   ),
                yaxis3 = dict( title='Test %s' % metrics,
                    titlefont = dict(color=cl3),
                    tickfont  = dict(color=cl3),
                    range     = ylim,
                    overlaying='y', side='right', position=0.98, anchor ='free'   )
            )


            fig = go.Figure(data=data, layout=layout)

            py.plot(fig, filename=fname, auto_open=False)


        def total_epoch_count(self):
            return self.epoch_count

#==============================================================================
# Done.
#==============================================================================
