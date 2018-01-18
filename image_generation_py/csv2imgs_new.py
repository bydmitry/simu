#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:38:38 2017

@author: bychkov
"""

import os
import skimage
import numpy as np

import sim_imgs

import numpy as np
import pandas as pd

#----------------------------------------------------------------------
# Settings:
#----------------------------------------------------------------------
set_name   = 'non-monotonic-risk'
save_to    = '/Users/bychkov/GDD/Projects/simu/synthetic_imgs/'

im_size    = (109, 109) # (111, 111)

csv_file = '/Users/bychkov/GDD/Projects/simu/simulated_data/survival_data_NS-3000_NC-3.csv'
#----------------------------------------------------------------------
# Read CSV data:
#----------------------------------------------------------------------
csv_data = pd.read_csv(csv_file)
csv_data = csv_data.head(5)

NS  = csv_data.shape[0]

del csv_file
#----------------------------------------------------------------------
# Prepare arrays:
#----------------------------------------------------------------------
print('Assembling Numpy arrays:')


# Initialize data arrays & indexes:
Xs  = np.empty(shape=[ NS, im_size[0], im_size[1], 3 ], dtype='float32')
Ys  = csv_data[['h2','t2','e2','bin_h1','bin_t2']].as_matrix()

for ind, row in csv_data.iterrows():
    print('  Sample: %d / %d' % ( ind+1, NS ))

    # Generate image:
    im = sim_imgs.sim_sample(canvas=im_size, xs=(row['ft1'].astype(np.int32),row['ft2'].astype(np.int32)), my_dpi=300.0)
    im = skimage.img_as_float( im )

    # Add Xs:
    Xs[ind,...] = im

del im, ind, row

#----------------------------------------------------------------------
# Dumping:
#----------------------------------------------------------------------
print('\nDumping...')

np.savez(
        os.path.join(save_to, set_name + '.npz'), # save_to
        Xs   = Xs,
        Ys   = Ys )

#----------------------------------------------------------------------
# Done.
#----------------------------------------------------------------------
