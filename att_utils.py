#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:00:38 2017

@author: bychkov
"""

import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
# Load syntetic data from npz archive
#----------------------------------------------------------------------
def load_synthetic_imgs(npz_path):
    data = np.load(npz_path)

    data_dict = {
        'train_x' : np.transpose(data['train_x'], (0,2,3,1)),
        'valid_x' : np.transpose(data['valid_x'], (0,2,3,1)),
        'test_x'  : np.transpose(data['test_x'], (0,2,3,1)),
        'train_y' : data['train_y'],
        'valid_y' : data['valid_y'],
        'test_y'  : data['test_y']
    }

    return data_dict

#----------------------------------------------------------------------
# Extract tiles
#----------------------------------------------------------------------
from sklearn.feature_extraction.image import extract_patches
#from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def extract_tiles(data_dict, w_size, s_size):
    for img_split in ['train_x','valid_x','test_x']:
        images = data_dict[img_split]

        N = images.shape[0]
        patches_lst = list()
        for p in xrange(N):
            tt = extract_patches(
                    images[p,...],
                    (w_size, w_size, 3), (s_size,s_size, 3) )
            patches_lst.append( np.squeeze(tt) )

        # Patch-grid dims
        grid2 = patches_lst[0].shape[0] * patches_lst[0].shape[1]
        # Assemble ndarray form list
        img_split += '_p'
        data_dict[img_split] = np.stack(patches_lst, axis=0)
        # Flatten patches
        data_dict[img_split] = np.reshape(
                data_dict[img_split],
                (N, grid2, w_size, w_size, 3) )

    return data_dict

#----------------------------------------------------------------------
# 2D-Image reconstruction from patches
#----------------------------------------------------------------------
from itertools import product

import scipy
import skimage

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#------------------------------------------------------------
def reconstruct_from_tiles_2d(patches, step, im_size):
    ''' Reconstruct 2D matricies from overlapping patches.

    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Parameters
    ----------
    '''
    # Grid size
    g_size   = int( np.sqrt(patches.shape[0]) )
    p_h, p_w = patches.shape[1:3]

    # Create new array of zeros:
    img = np.zeros(im_size, dtype = patches.dtype )

    # This is to iterate over tile masks:
    for patch, (r, c) in zip(patches, product(range(g_size),range(g_size))):
        #print patch.shape, r, c
        ir = step * r;  ic = step * c
        img[ir:ir + p_h, ic:ic + p_w] = patch
    return img

#### Testing: ###
#one_image = np.arange(111 * 111).reshape((111, 111))
#patches = extract_patches(one_image, (40, 40), (35,35))
#print patches.shape
#patches = np.reshape(patches, (3*3, 40, 40) )
#print patches.shape
#patches = reconstruct_from_tiles_2d(patches, 35, (111,111))
#print patches.shape
#patches
#
#np.testing.assert_array_equal(one_image[:-1,:-1], patches[:-1,:-1])
#----------------------------------------------------------------------
# Upsacel tiles:
#----------------------------------------------------------------------
def upscale_mask_tiles(masks, new_size, rescale=True):
    '''
    masks : ndarray, (num_tiles, height, width)
    new_sie : tuple of ints, (new_height, new_width)
    '''
    num_tiles = masks.shape[0]
    # Create new array of zeros:
    upscaled_msk =  np.zeros(
            (num_tiles,new_size[0],new_size[1]),
            dtype = masks.dtype )

    # Iterate over patches:
    for p in xrange(num_tiles):
        tmp = masks[p]
        if rescale:
            tmp = skimage.img_as_ubyte(tmp)
        upscaled_msk[p] = scipy.misc.imresize(
                tmp, new_size )

    return upscaled_msk

#----------------------------------------------------------------------
# Done.
#----------------------------------------------------------------------