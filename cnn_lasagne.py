#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:04:05 2017

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

import myutils

#----------------------------------------------------------------------
# Settings:
#----------------------------------------------------------------------
data = np.load('synthetic_imgs/first.npz')
class_names = np.array(['benign','cancer'])

train_x = data['train_x']
train_y = data['train_y']
test_x  = data['test_x']
test_y  = data['test_y']

#Xs = np.concatenate((test_x, train_x), axis=0)
#Ys = np.concatenate((test_y, train_y), axis=0)
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
# Define network architecture:
#----------------------------------------------------------------------
input_var   = T.tensor4('inputs')
target_var  = T.ivector('targets')

#---  Network  --- #
net  = ll.InputLayer( shape=(None, 3, 90, 90), input_var=input_var )

net  = ll.Conv2DLayer( net, 32, (3,3), nonlinearity=ln.sigmoid ) 
net  = ll.Pool2DLayer( net, pool_size=(2,2) )

net  = ll.Conv2DLayer( net, 32, (3,3), nonlinearity=ln.sigmoid ) 
net  = ll.Pool2DLayer( net, pool_size=(2,2) )

net  = ll.Conv2DLayer( net, 64, (3,3), nonlinearity=ln.sigmoid ) 
net  = ll.Pool2DLayer( net, pool_size=(2,2) )

net  = ll.DenseLayer( net, 64, nonlinearity=ln.sigmoid)
net  = ll.DenseLayer( net, 1, nonlinearity=ln.sigmoid)
net  = ll.ReshapeLayer( net, (-1,) )

# Predictions:
probs = ll.get_output( net, deterministic=True )
preds = probs > 0.5

#---  Losses  --- #
loss  = binary_crossentropy(probs, target_var).mean()
acc   = T.mean(T.eq(preds, target_var), dtype='float32')

#---  Training settings  --- #
params   = ll.get_all_params(net, trainable=True)
updates  = lasagne.updates.adadelta(loss, params)

#---  Compile Theano functions  --- #
train_fn  = theano.function([input_var, target_var], [loss, acc], updates=updates)    
valid_fn  = theano.function([input_var, target_var], [loss, acc])


#---  Training Loop  --- #
num_epochs  = 100
batch_size  = 10

for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_acc = 0
    train_batches = 0
    start_time = time.time()
    
    for batch in myutils.iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = train_fn(inputs, targets)
        train_err += err
        train_acc += acc
        train_batches += 1

    # And a full pass over the validatio data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in myutils.iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
        inputs, targets = batch
        v_err, v_acc = valid_fn(inputs, targets)
        val_err += v_err
        val_acc += v_acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.2f} %".format(
        train_acc / train_batches * 100))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

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