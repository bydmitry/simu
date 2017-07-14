#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:33:10 2017

@author: bychkov
"""

import os
import skimage
import numpy as np

import sim_imgs


import numpy as np

#==============================================================================
# A class to render toy tissue images.
#==============================================================================

class KudosMaker(object):
    def __init__(self):
        pass
    
#==============================================================================
# Done.
#==============================================================================


#----------------------------------------------------------------------
# Settings:
#----------------------------------------------------------------------
set_name   = 'first'
save_to    = 'synthetic_imgs'
dump_png   = False
dump_npz   = True

trn_smpl   = 1000
tst_smpl   = 300

im_size    = 90
normalize  = False


#----------------------------------------------------------------------
# Prepare arrays:
#----------------------------------------------------------------------
print 'Assembling Numpy arrays:'
# Initialize data arrays:
train_x  = np.empty(shape=[ trn_smpl, 3, im_size, im_size ], dtype='float32')
train_y  = np.empty(shape=[ trn_smpl ], dtype='uint8')
train_id = np.empty(shape=[ trn_smpl ], dtype='int32')

test_x   = np.empty(shape=[ tst_smpl, 3, im_size, im_size ], dtype='float32')
test_y   = np.empty(shape=[ tst_smpl ], dtype='uint8')
test_id  = np.empty(shape=[ tst_smpl ], dtype='int32')

# Looping through TRAIN samples:
for ind in range(trn_smpl):
    print '  Train sample: %d / %d' % ( ind+1, trn_smpl )

    # Prepare full and masked images:
    im = sim_imgs.sim_sample(cancer_cells = bool(ind%2))

    # Convert to float:
    im = skimage.img_as_float( im )

    # Add to numpy array:
    train_x[ind,...] = np.transpose( im, axes = (2,0,1) )

    # Add Y (target class) and Id:
    train_y[ind]     = int( ind%2 )      # Y
    train_id[ind]    = int( ind )        # ID

# Looping through TEST samples:
for ind in range(tst_smpl):
    print '  Train sample: %d / %d' % ( ind+1, tst_smpl )

    # Prepare full and masked images:
    im = sim_imgs.sim_sample(cancer_cells = bool(ind%2))

    # Convert to float:
    im = skimage.img_as_float( im )

    # Add to numpy array:
    test_x[ind,...] = np.transpose( im, axes = (2,0,1) )

    # Add Y (target class) and Id:
    test_y[ind]     = int( ind%2 )      # Y
    test_id[ind]    = int( ind )        # ID

del im, ind
#----------------------------------------------------------------------
# Normalize:
#----------------------------------------------------------------------
if normalize:
    print '\nNormalization...'
    # Normalization and scaling based on training statistics
    tr_set_mean = np.mean( train_x )
    tr_set_std  = np.std( train_x )

    train_x = train_x - tr_set_mean
    #train_x = skimage.exposure.rescale_intensity(train_x, in_range='float32')
    #train_x = train_x / 256.     #/ tr_set_std

    test_x  = test_x  - tr_set_mean
    #test_x = skimage.exposure.rescale_intensity(test_x, in_range='float32')
    #test_x  = test_x  / 256.     # / tr_set_std
else:
    print '\nNo normalization...'
    tr_set_mean = 0.
    tr_set_std  = 0.

    #train_x = train_x / 256.
    #test_x  = test_x  / 256.

#----------------------------------------------------------------------
# Dumping:
#----------------------------------------------------------------------
print '\nDumping...'

np.savez(
        os.path.join(save_to, set_name + '.npz'), # save_to
        train_x   = train_x,
        train_y   = train_y,
        train_id  = train_id,
        test_x    = test_x,
        test_y    = test_y,
        test_id   = test_id )

#----------------------------------------------------------------------
# Done.
#----------------------------------------------------------------------