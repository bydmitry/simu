#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:04:22 2017

@author: bychkov
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
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
# Load Survival Samples:
#----------------------------------------------------------------------
def load_surv_samples(fname, sort=False):
    dataset  = pickle.load( open( fname, "rb" ) )
    if not isinstance(dataset, dict):
        print('Cannot load the data!')
        return False

    if sort:
        # Sort Training Data for Accurate Likelihood
        sort_idx = np.argsort(dataset['t'])[::-1]
        for k in dataset.keys():
            if type(dataset[k]) == np.ndarray:
                if dataset[k].shape[0] == dataset['t'].shape[0]:
                   dataset[k] =  dataset[k][sort_idx]

    # (N, d)
    data_x  = dataset['x']

    # (N, 3)
    data_y  = np.column_stack((
                    dataset['h'],
                    dataset['t'],
                    dataset['e'],
                    dataset['id'] ))


    print('Loading data from: %s' % fname)
    return (data_x, data_y)
#----------------------------------------------------------------------
# Losses:
#----------------------------------------------------------------------
def efron_estimator_tf(y_true, y_pred):
    sort_idx = tf.nn.top_k(y_true[:,1], k=tf.shape(y_pred)[0], sorted=True).indices

    risk          = tf.gather(y_pred, sort_idx)
    risk_exp      = tf.exp(risk)
    events        = tf.gather(y_true[:,2], sort_idx)
    ftimes        = tf.gather(y_true[:,1], sort_idx)
    ftimes_cens   = ftimes * events

    # Get unique failure times & Exclude zeros
    # NOTE: this assumes that falure times start from > 0 (greater than zero)
    unique = tf.unique(ftimes_cens).y
    unique_ftimes = tf.boolean_mask(unique, tf.greater(unique, 0) )
    m = tf.shape(unique_ftimes)[0]

    # Define key variables:
    log_lik  = tf.Variable(0., dtype=tf.float32, validate_shape=True, trainable=False)
    E_ti     = tf.Variable([], dtype=tf.int32,   validate_shape=True, trainable=False)
    risk_phi = tf.Variable([], dtype=tf.float32, validate_shape=True, trainable=False)
    tie_phi  = tf.Variable([], dtype=tf.float32, validate_shape=True, trainable=False)
    cum_risk = tf.Variable([], dtype=tf.float32, validate_shape=True, trainable=False)
    cum_sum  = tf.cumsum(risk_exp)

    # -----------------------------------------------------------------
    # Prepare for looping:
    # -----------------------------------------------------------------
    i = tf.constant(0, tf.int32)
    def loop_cond(i, *args):
        return i < m

    # Step for loop # 1:
    def loop_1_step(i, E, Rp, Tp, Cr, Cs):
        n = tf.shape(Cs)[0]
        idx_b = tf.logical_and(
            tf.equal(ftimes, unique_ftimes[i]),
            tf.equal(events, tf.ones_like(events)) )

        idx_i = tf.cast(
            tf.boolean_mask(
                tf.lin_space(0., tf.cast(n-1,tf.float32), n),
                tf.greater(tf.cast(idx_b, tf.int32),0)
            ), tf.int32 )

        E  = tf.concat([E, [tf.reduce_sum(tf.cast(idx_b, tf.int32))]], 0)
        Rp = tf.concat([Rp, [tf.reduce_sum(tf.gather(risk, idx_i))]], 0)
        Tp = tf.concat([Tp, [tf.reduce_sum(tf.gather(risk_exp, idx_i))]], 0)

        idx_i = tf.cast(
            tf.boolean_mask(
                tf.lin_space(0., tf.cast(n-1,tf.float32), n),
                tf.greater(tf.cast(tf.equal(ftimes, unique_ftimes[i]), tf.int32),0)
            ), tf.int32 )

        Cr = tf.concat([Cr, [tf.reduce_max(tf.gather( Cs, idx_i))]], 0)
        return i + 1, E, Rp, Tp, Cr, Cs

    # Step for loop # 1:
    def loop_2_step(i, E, Rp, Tp, Cr, likelihood):
        l = E_ti[i]
        J = tf.lin_space(0., tf.cast(l-1,tf.float32), l) / tf.cast(l, tf.float32)
        Dm = Cr[i] - J * Tp[i]
        likelihood = likelihood + Rp[i] - tf.reduce_sum(tf.log(Dm))
        return i + 1, E, Rp, Tp, Cr, likelihood

    # -----------------------------------------------------------------

    # Loop # 1:
    _, E_ti, risk_phi, tie_phi, cum_risk, _ = loop_1 = tf.while_loop(
        loop_cond, loop_1_step,
        loop_vars = [i, E_ti, risk_phi, tie_phi, cum_risk, cum_sum],
        shape_invariants = [i.get_shape(),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),cum_sum.get_shape()]
    )

    # Loop # 2:
    loop_2 = tf.while_loop(
        loop_cond, loop_2_step,
        loop_vars = [i, E_ti, risk_phi, tie_phi, cum_risk, log_lik],
        shape_invariants = [i.get_shape(),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None]),log_lik.get_shape()]
    )

    log_lik = loop_2[5]
    return tf.negative(log_lik)

def my_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true[:,0]))

# def partial_likelihood(y_true, y_pred):
#     ''' Returns the Negative Partial Log Likelihood
#         of the parameters given ordered hazards [estimated by a NN];
#         This method cannot handle tied observations.
#
#         Parameters:
#             risk (N,): a vector of hazard values for each of N samples.
#                 Samples (=> risk vector) are ordered according to failure
#                 time from largest to smallest.
#             events (N,): a vector (ordered in the same fashion) of event
#                 indicators, where 1 - is event; 0 - is censored.
#     '''
#     #y_true = theano.tensor.fmatrix()
#     #y_pred = theano.tensor.fvector()
#
#     # first sort by time for Accurate Likelihood:
#     sort_idx = np.argsort( y_true[:,1] )[::-1]
#
#     risk    = y_pred[sort_idx]
#     events  = y_true[:,2][sort_idx]
#
#     hazard_ratio = T.exp(risk)
#     log_cum_risk = T.log(T.extra_ops.cumsum(hazard_ratio))
#     uncencored_likelihood = risk.T - log_cum_risk
#     censored_likelihood = uncencored_likelihood * events
#     neg_likelihood = -T.sum( censored_likelihood )
#
#     return neg_likelihood

#----------------------------------------------------------------------
# Done.
#----------------------------------------------------------------------
