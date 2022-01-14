#!/usr/bin/env python
#   Copyright (C) 2016 University of Oxford 
#   SHBASECOPYRIGHT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
from utils import *
from scipy.ndimage import filters
import math
import scipy
from skimage import exposure
import augmentations_gmdist
import augmentations_distmaps2, augmentations_distmaps
from skimage.transform import resize

def preprocess_data(data):    
    return (2 * data / np.amax(np.reshape(data,[-1,1]))) - 1

def preprocess_data_gauss(data):
    brain1 = data > 0
    brain = brain1 > 0
    data = data - np.mean(data[brain])      
    den = np.std(data[brain])
    if den == 0:
        den = 1
    data = data/den
    data[brain==0] = np.min(data)
    return data

def rescale_its_intensity(image):
    min_limit = 50
    v_min, v_max = np.percentile(image, (min_limit, 100))
    new_img = exposure.rescale_intensity(image, in_range=(v_min, v_max))    
    return new_img

def cut_zeros1d(im_array):
    im_list = list(im_array > 0)
    start_index = im_list.index(1)
    end_index = im_list[::-1].index(1)
    length = len(im_array[start_index:])-end_index
    return start_index, end_index, length

def tight_crop_data(img_data):
    row_sum = np.sum(np.sum(img_data,axis=1),axis=1)
    col_sum = np.sum(np.sum(img_data,axis=0),axis=1)
    stack_sum = np.sum(np.sum(img_data,axis=1),axis=0)
    rsid, reid, rlen = cut_zeros1d(row_sum)
    csid, ceid, clen = cut_zeros1d(col_sum)
    ssid, seid, slen = cut_zeros1d(stack_sum)
    return img_data[rsid:rsid+rlen, csid:csid+clen, ssid:ssid+slen], [rsid, rlen, csid, clen, ssid, slen]

def get_slice_weights(brain_sub_piece):
    prslices = -2 + 0*brain_sub_piece
    n_rows_top = brain_sub_piece.shape[2]//3 + np.floor(brain_sub_piece.shape[2] * 0.45).astype(int)
    n_rows_bot = brain_sub_piece.shape[2]//3
    #print(n_rows_top, n_rows_bot)
    prslices[:,:,n_rows_bot:n_rows_top] = 1
    slwei = scipy.ndimage.morphology.distance_transform_edt(prslices>0)
    slwei = slwei - np.mean(slwei[brain_sub_piece == 1])
    den = np.std(slwei[brain_sub_piece == 1])
    slwei = (slwei/den) * brain_sub_piece
    slwei[brain_sub_piece == 0] = np.amin(slwei)
    return slwei 

def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
            warnings.warn("Only grayscale images allowed, converting to 2D matrix")
            img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
            import pylab as pl
            from time import sleep

            fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
            ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

            ax1.imshow(img,interpolation='nearest')
            ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
            ax1.set_title("Original image")
            ax2.set_title("Iteration 0")

            fig.canvas.draw()

    for ii in range(niter):

            # calculate the diffs
            deltaS[:-1,: ] = np.diff(imgout,axis=0)
            deltaE[: ,:-1] = np.diff(imgout,axis=1)

            # conduction gradients (only need to compute one per dim!)
            if option == 1:
                    gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                    gE = np.exp(-(deltaE/kappa)**2.)/step[1]
            elif option == 2:
                    gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                    gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

            # update matrices
            E = gE*deltaE
            S = gS*deltaS

            # subtract a copy that has been shifted 'North/West' by one
            # pixel. don't as questions. just do it. trust me.
            NS[:] = S
            EW[:] = E
            NS[1:,:] -= S[:-1,:]
            EW[:,1:] -= E[:,:-1]

            # update the image
            imgout += gamma*(NS+EW)

            if ploton:
                    iterstring = "Iteration %i" %(ii+1)
                    ih.set_data(imgout)
                    ax2.set_title(iterstring)
                    fig.canvas.draw()
                    # sleep(0.01)

    return imgout

def histogram_equalisation(data1):
    data2 = exposure.equalize_hist(data1[data1>0])
    #print(data1.shape)
    if len(data1.shape)==3:
        data = np.zeros([data1.shape[0], data1.shape[1], data1.shape[2]])
    else:
        data = np.zeros([data1.shape[0], data1.shape[1], data1.shape[2], data1.shape[3]])
    data[data1>0] = data2
    return data
