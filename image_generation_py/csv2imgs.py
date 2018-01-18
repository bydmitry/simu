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
set_name   = 'test'
save_to    = '/Users/bychkov/GDD/Projects/simu/synthetic_imgs/'

im_size    = (109, 109) # (111, 111)

csv_file = '/Users/bychkov/GDD/Projects/simu/simulated_data/linear_5K.csv'
csv_splits = '/Users/bychkov/GDD/Projects/simu/models/MLP_3hu_linear/splits.csv'

#----------------------------------------------------------------------
# OldRange = (OldMax - OldMin)
OldRange = ( 1 - (-1) )

# NewRange = (NewMax - NewMin)
x1Range = (200 - 0)
x2Range = (30 - 0)

#NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
#----------------------------------------------------------------------
# Read CSV data:
#----------------------------------------------------------------------
csv_data = pd.DataFrame.merge(
    pd.read_csv(csv_file),
    pd.read_csv(csv_splits),
    on = 'id'
)

n_smpl = dict()
n_smpl['train'] = sum(csv_data['split'] == 'train')
n_smpl['valid'] = sum(csv_data['split'] == 'valid')
n_smpl['test']  = sum(csv_data['split'] == 'test')
n_smpl['N']     = csv_data.shape[0]

del csv_splits, csv_file
#----------------------------------------------------------------------
# Prepare arrays:
#----------------------------------------------------------------------
print 'Assembling Numpy arrays:'

np_dict = dict(); np_indx = dict();
# Initialize data arrays & indexes:
for split in ['train', 'valid', 'test']:
    np_dict[split+'_x'] = np.empty(shape=[ n_smpl[split], 3, im_size[0], im_size[1] ], dtype='float32')
    np_dict[split+'_y'] = np.empty(shape=[ n_smpl[split], 4], dtype='float32')
    np_indx[ split ]    = 0

for ind, row in csv_data.iterrows():
    print '  Sample: %d / %d' % ( ind+1, n_smpl['N'] )

    cur_split = row['split']

    # Prepare covariates:
    x1 = int(round( (((row['x.1'] + 1) * x1Range) / OldRange) ))
    x2 = int(round( (((row['x.2'] + 1) * x2Range) / OldRange) ))

    # Generate image:
    im = sim_imgs.sim_sample(canvas=im_size, xs=(x2,x1), my_dpi=300.0)
    im = skimage.img_as_float( im )

    # Add Xs:
    np_dict[cur_split+'_x'][np_indx[cur_split],...] = np.transpose(im, axes = (2,0,1))

    # Add Ys:
    np_dict[cur_split+'_y'][np_indx[cur_split],0] = row['h']
    np_dict[cur_split+'_y'][np_indx[cur_split],1] = row['t']
    np_dict[cur_split+'_y'][np_indx[cur_split],2] = row['e']
    np_dict[cur_split+'_y'][np_indx[cur_split],3] = row['id']

    # Do not forget to increment counters:
    np_indx[cur_split] += 1

del im, ind, row, x1, x2

#----------------------------------------------------------------------
# Dumping:
#----------------------------------------------------------------------
print '\nDumping...'

np.savez(
        os.path.join(save_to, set_name + '.npz'), # save_to
        train_x   = np_dict['train_x'],
        train_y   = np_dict['train_y'],
        valid_x   = np_dict['valid_x'],
        valid_y   = np_dict['valid_y'],
        test_x    = np_dict['test_x'],
        test_y    = np_dict['test_y'] )

#----------------------------------------------------------------------
# Done.
#----------------------------------------------------------------------
