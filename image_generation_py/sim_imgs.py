#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:36:01 2017

@author: bychkov
"""

import math
import random
import numpy as np

from skimage import exposure
from scipy import ndimage, misc

import matplotlib as mplib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#plt.ioff()

from opensimplex import OpenSimplex

#==============================================================================
# Simulate background with Perlin Noise:
# [http://physbam.stanford.edu/cs448x/old/Procedural_Noise%282f%29Perlin_Noise.html ]
#==============================================================================
def turbulence(pos, pixel_size):
    gen = OpenSimplex(random.randrange(1,1e5))
    
    x = 0.0
    scale = 1.0;
    while( scale > pixel_size):
        pos = pos/scale
        x = x + gen.noise2d(pos[0],pos[1])*scale
        scale = scale/2.0
    return x


def marble(pos):
    rgb = np.zeros((3))
    x = np.sin( (pos[1] + 3.0 * turbulence(pos,0.025)) * np.pi )
    x = np.sqrt(x+1) * 0.71
    rgb[1] = 0.2 + 0.8 * x
    rgb[0] = 0.3 + 0.6 * np.sqrt(x)
    rgb[2] = 0.6 + 0.4 * np.sqrt(x)
    return rgb

#------------------------------------------------------------------------------  
  
def sim_background(grid=8, im_resize=(90,90), smooth=11): 
    im_resize = im_resize + (3,)
    # gen.noise3d(x, y, c) / 2.0 + 0.5
    h = w = grid
    
    simg = np.zeros((h,w,3), dtype='float32')
    for y in xrange(h):
        for x in xrange(w):
            simg[y][x] = marble(np.array([x,y])) 
                
    # Resize:
    simg = misc.imresize(simg, size=im_resize)
    
    # Smoothing:
    for c in range(3):
        simg[:,:,c] = ndimage.uniform_filter(simg[:,:,c],size=smooth)
    
    #plt.imshow(simg)
    return simg

#==============================================================================
# Foreground objects:
#==============================================================================
def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ) :
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (float(x),float(y)) )

        angle = angle + angleSteps[i]

    return tuple(points)

def clip(x, min, max):
     if( min > max ) :  return x    
     elif( x < min ) :  return min
     elif( x > max ) :  return max
     else :             return x

def cell_pathces_1(n_cells, canvas_size):
    Path = mpath.Path
    #clump_mrgn = np.round(canvas_size[0]*0.15)
    
    # This setting is good for n_cells = [0,30]
    averRad = 4
    clump_mrgn = np.round(averRad*0.8*n_cells*0.3 + 5)
    
    # Clump location:
    cntr  = np.random.randint(0+clump_mrgn,89-clump_mrgn,(2))    
    spanX = np.random.randint(
            cntr[0]-clump_mrgn, cntr[0]+clump_mrgn,(n_cells) )
    spanY = np.random.randint(
            cntr[1]-clump_mrgn, cntr[1]+clump_mrgn,(n_cells) )
        
    patches = list()
    # Tuple of codes:
    codes = (Path.MOVETO,) 
    for i in [Path.CURVE4 for c in xrange(9)]:
        codes = codes + (i,)
        
    # --- Generate Vertices --- #
    for cc_n in xrange(n_cells):
        verts = generatePolygon(ctrX=spanX[cc_n], ctrY=spanY[cc_n], numVerts=9,
                                aveRadius=averRad, irregularity=0.5, spikeyness=0.1)
        # Last point is same as first to close the polygon
        verts = verts + (verts[0],)    
            
        # Create a patch:
        path   = mpath.Path(verts, codes)
        patch  = mpatches.PathPatch(path, 
                        facecolor='r', alpha=0.39,
                        lw = 0.35)
        patches.append( patch )    
    
    return patches

def cell_pathces_2(n_cells, canvas_size):
    patches = list()
    
    spanX = np.random.randint( 0, canvas_size[0], n_cells )
    spanY = np.random.randint( 0, canvas_size[1], n_cells )

    for cc_n in xrange(n_cells):
        patch = mpatches.Circle(
            xy         = (spanX[cc_n],spanY[cc_n]),
            radius     = int(np.round(canvas_size[0]*0.01)),
            facecolor  = '#000066',
            edgecolor  = 'none',
            alpha      = 0.7
        )
        patches.append(patch)
    
    return patches

#==============================================================================
# Simulate sample function:
#==============================================================================   
def sim_sample(canvas=(90,90), xs=(5,50), my_dpi=300.0):
    '''
    my_dpi: Must be float!
    
    '''
    # Initialize figure:
    fig, ax = plt.subplots(figsize=(canvas[0]/my_dpi,canvas[1]/my_dpi), dpi=my_dpi)

    # --- Add background to the figure --- #
    grid=int(canvas[0]/6.0); smooth=canvas[0]/4.5;
    bckgrd = sim_background(grid, canvas, smooth)
    ax.imshow( bckgrd )
    
    # --- Add patches to the figure --- #
    patches = cell_pathces_1(n_cells=xs[0], canvas_size=canvas)
    for patch in patches:
        ax.add_patch(patch)
        
    patches = cell_pathces_2(n_cells=xs[1], canvas_size=canvas)
    for patch in patches:
        ax.add_patch(patch)
        
    # --- Adjust Axis & Margins --- #
    ax.axis('equal')
    ax.set_ylim( [0, canvas[0]] )
    ax.set_xlim( [0, canvas[1]] )
    ax.axis('off')
    ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(bottom=0, right=1, top=1, left=0)
    
    # plot control points and connecting lines
    #x, y = zip(*path.vertices)
    #line, = ax.plot(x, y, 'go-')
    
    # --- Convert figure to Numpy Arr --- #
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h   = fig.canvas.get_width_height()
    buf    = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( h, w, 4 )
     
    # canvas.tostring_argb give pixmap in ARGB mode. 
    # Roll the ALPHA channel to have it in RGBA mode.
    buf = np.roll(buf, 3, axis = 2)
    
    # Save
    #misc.toimage(buf, cmin=0.0, cmax=255.0).save('test_imgs/sample.jpg')
    
    plt.clf()
    plt.close()
    
    return buf[:,:,0:3]

#==============================================================================
# Simulate synthetic images:
#==============================================================================

#smpl   = sim_sample(cancer_cells=True)




#==============================================================================
# Done.
#==============================================================================