#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:01:10 2017

@author: bychkov
"""
import time
import numpy as np

import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as ln
from lasagne.objectives import binary_crossentropy

import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import nolearn.lasagne as nolas
from nolearn.lasagne import TrainSplit, BatchIterator

import sklearn
from sklearn.metrics import roc_auc_score
import myutils

#----------------------------------------------------------------------
# Settings:
#----------------------------------------------------------------------
data = np.load('synthetic_imgs/first.npz')
class_names = np.array(['benign','cancer'])

train_x = data['train_x']
train_y = data['train_y'].reshape((-1,1))
test_x  = data['test_x']
test_y  = data['test_y'].reshape((-1,1))

#Xs = np.concatenate((test_x, train_x), axis=0)
#Ys = np.concatenate((test_y, train_y), axis=0).astype(np.int32).reshape((-1,1))
#----------------------------------------------------------------------
# Visualize some data:
#----------------------------------------------------------------------

plt.figure(figsize=[6,5])
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.xlabel(class_names[train_y[i]])
    plt.imshow(np.transpose(train_x[i],[1,2,0]))

plt.close()
del i

#----------------------------------------------------------------------
# Configure model:
#----------------------------------------------------------------------

#---  Network  --- #
layers_lst = [
    # Input:
    (ll.InputLayer, {'shape': (None,3,90,90) }),
    
    # Convolutions:
    (ll.Conv2DLayer, {'num_filters': 32, 'filter_size': (3,3)}),
    (ll.Pool2DLayer, {'pool_size': (2,2)}),

    (ll.Conv2DLayer, {'num_filters': 32, 'filter_size': (3,3)}),
    (ll.Pool2DLayer, {'pool_size': (2,2)}),

    (ll.Conv2DLayer, {'num_filters': 64, 'filter_size': (3,3)}),
    (ll.Pool2DLayer, {'pool_size': (2,2)}),

    (ll.DenseLayer, {'num_units': 64}),
    (ll.DenseLayer, {'num_units': 1, 'nonlinearity': ln.sigmoid}),
    
    # Dimention tuning:
    (ll.ReshapeLayer, {'shape': (-1,1)}),
]

    
#---  Initialise nolearn NN object  --- #   
net_cnn = nolas.NeuralNet(
    layers      = layers_lst,
        
    # Optimization:
    max_epochs  = 10,
    update      = lasagne.updates.adadelta,
    
    # Objective:
    objective_loss_function = lasagne.objectives.binary_crossentropy,
    
    # Batch size & Splits:
    train_split           = TrainSplit( eval_size=.3 ),
    batch_iterator_train  = BatchIterator(batch_size=10, shuffle=False),
    batch_iterator_test   = BatchIterator(batch_size=10, shuffle=False),
    
    # Custom scores:
    # 1) target; 2) preds:
    custom_scores  = [('auc', lambda y_true, y_proba: roc_auc_score(y_true, y_proba[:,0]))],
    # 1) preds; 2) target;
    scores_train   = None,
    scores_valid   = None, 
    
    # misc:
    y_tensor_type  = T.imatrix,
    regression     = True,
    verbose        = 1,
    
    # CallBacks:    
    on_training_started   = None,
    on_training_finished  = None,
    on_batch_finished     = None,
    on_epoch_finished     = [ myutils.PlotLosses(figsize=(8,6)) ],
)

#----------------------------------------------------------------------
# Train:
#----------------------------------------------------------------------
net_cnn.fit(train_x, train_y)

#----------------------------------------------------------------------
# Evaluate:
#----------------------------------------------------------------------
proba = net_cnn.predict_proba(test_x)

auc  = sklearn.metrics.roc_auc_score(test_y[:,0], proba[:,0])


#----------------------------------------------------------------------
# ...
#----------------------------------------------------------------------
# Optionally, you could now dump the network weights to a file like this:
# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)



#----------------------------------------------------------------------
# Done.
#----------------------------------------------------------------------