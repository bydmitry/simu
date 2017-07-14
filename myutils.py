#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:04:22 2017

@author: bychkov
"""

import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
# Plot losses:
#----------------------------------------------------------------------
class PlotLosses(object):
    def __init__(self, figsize=(8,6)):
        plt.plot([], []) 

    def __call__(self, nn, train_history):
        train_loss = np.array([i["train_loss"] for i in nn.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in nn.train_history_])

        plt.gca().cla()
        plt.plot(train_loss, label="train") 
        plt.plot(valid_loss, label="test")

        plt.grid()
        plt.legend()
        plt.draw()

#----------------------------------------------------------------------
# Batch iterator:
#----------------------------------------------------------------------
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
#----------------------------------------------------------------------
# Done.
#----------------------------------------------------------------------        