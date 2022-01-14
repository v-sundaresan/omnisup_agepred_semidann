#!/usr/bin/env python
#   Copyright (C) 2016 University of Oxford 
#   SHBASECOPYRIGHT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from utils import *
from scipy.ndimage import filters
import math
from skimage import exposure
import augmentations_gmdist
import augmentations_distmaps2, augmentations_distmaps
from skimage.transform import resize
import data_preprocessing
from skimage.morphology import dilation
from skimage.measure import regionprops, label
from skimage import morphology
from skimage import measure
import nibabel as nib


def patches_to_volumes(patches4d,volume3d,patch_size=64):
    volume4d = np.zeros([volume3d.shape[0],volume3d.shape[1],volume3d.shape[2],patches4d.shape[3]])
    i = 0   
    for s in range(stackcount):
        for r in range(rowcount):
            for c in range(colcount):
                start_row = r*patch_size,
                start_col = c*patch_size
                start_stack = s*patch_size
                end_row = np.amin([(r+1)*patch_size,volume3d.shape[0]])
                end_col = np.amin([(c+1)*patch_size,volume3d.shape[1]])
                end_stack = np.amin([(s+1)*patch_size,volume3d.shape[2]])
                crop_patch = volume3d[start_row:end_row, start_col:end_col, start_stack:end_stack]
                patch_dim = crop_patch.shape
                volume4d[start_row:end_row, start_col:end_col, start_stack:end_stack,:] = patches4d[i,:patch_dim[0], :patch_dim[1], :patch_dim[2],:]
                i += 1
    return volume4d


def construct_image_from_patches(filenames,patches5d,patch_size=64):
    patches5d = patches5d.transpose(0,2,3,1,4)
    volume5d = []
    for num_images in range(len(filenames)):
        mc = np.load(filenames[num_images])
        data_sub = mc['data']
        _,coords = data_preprocessing.tight_crop_data(data_sub)
        volume = data_sub[coords[0]:coords[0]+coords[1],coords[2]:coords[2]+coords[3],coords[4]:coords[4]+coords[5]]
        rowcount = math.ceil(volume.shape[0]/patch_size)
        colcount = math.ceil(volume.shape[1]/patch_size)
        stackcount = math.ceil(volume.shape[2]/patch_size)
        count = 0
        for s in range(stackcount):
            for r in range(rowcount):
                for c in range(colcount):
                    count += 1
        patches4d = patches5d[num_images*count:(num_images+1)*count,:,:,:,:]
        volume4d = patches_to_volumes(patches4d,volume)
        volume5d.append(volume4d)
    return volume5d


def patches_from_local_maxima_to_volume(patches4d, volume3d, patch_size=48):
    volume4d = np.zeros([volume3d.shape[0],volume3d.shape[1],volume3d.shape[2],patches4d.shape[-1]])
    test_img1 = data_preprocessing.preprocess_data(volume3d)
    test_img = filters.gaussian_filter(test_img1, 0.4)
    loc_maxima = morphology.h_maxima(test_img,0.2)
    labs,num_labs = morphology.label(loc_maxima,return_num=True)
    props = measure.regionprops(labs)
    cent = np.array([prop.centroid for prop in props])
    int_range = []
    for c in range(cent.shape[0]):
        int_range.append(test_img[int(cent[c,0]),int(cent[c,1]),int(cent[c,2])])
    int_array = np.array(int_range) 
    perc = np.percentile(int_array, 80)
    deep_lesions = np.where(int_array <= perc)[0]
    cent = cent[deep_lesions,:]
    i = 0
    for c in range(cent.shape[0]):
        start_row = (np.amax([cent[c,0] - patch_size//2, 0])).astype(int)
        start_col = (np.amax([cent[c,1] - patch_size//2, 0])).astype(int)
        start_stack = (np.amax([cent[c,2] - patch_size//2, 0])).astype(int)
        end_row = (np.amin([cent[c,0] + patch_size//2,volume3d.shape[0]])).astype(int)
        end_col = (np.amin([cent[c,1] + patch_size//2,volume3d.shape[1]])).astype(int)
        end_stack = (np.amin([cent[c,2] + patch_size//2,volume3d.shape[2]])).astype(int)
        crop_patch = volume3d[start_row:end_row, start_col:end_col, start_stack:end_stack]
        patch_dim = crop_patch.shape
        volume4d[start_row:end_row, start_col:end_col, start_stack:end_stack,:] = patches4d[i,:patch_dim[0], :patch_dim[1], :patch_dim[2],:]
        i += 1
    return volume4d


def construct_image_from_local_maxima_patches(filenames,patches5d,patch_size=48):
    patches5d = patches5d.transpose(0,2,3,1,4)
    volume5d = []
    count_end_old = 0
    for num_images in range(len(filenames)):
        mc = np.load(filenames[num_images])
        data_sub = mc['data']
        _,coords = data_preprocessing.tight_crop_data(data_sub)
        volume = data_sub[coords[0]:coords[0]+coords[1],coords[2]:coords[2]+coords[3],coords[4]:coords[4]+coords[5]]
        test_img = data_preprocessing.preprocess_data(volume)
        test_img = filters.gaussian_filter(test_img, 0.4)
        loc_maxima = morphology.h_maxima(test_img,0.2)
        labs,num_labs = morphology.label(loc_maxima,return_num=True)
        count_end = count_end_old + num_labs
        count_start = count_end_old        
        patches4d = patches5d[count_start:count_end,:,:,:,:]
        volume4d = patches_from_local_maxima_to_volume(patches4d,volume)
        volume5d.append(volume4d)
    return volume5d

                    
def patches2D_from_local_maxima_to_volume(patches4d, volume3d, patch_size=64):
    volume4d = np.zeros([volume3d.shape[0],volume3d.shape[1],patches4d.shape[-1],volume3d.shape[2]])
    test_img1 = data_preprocessing.preprocess_data(volume3d)
    test_img = filters.gaussian_filter(test_img1, 0.4)
    #loc_maxima = morphology.h_maxima(test_img,0.2)
    for st in range(test_img1.shape[2]):
        loc_maxima = morphology.h_maxima(test_img[:,:,st],0.2)
        labs,num_labs = morphology.label(loc_maxima,return_num=True)   
        props = measure.regionprops(labs)
        cent = np.array([prop.centroid for prop in props])
        int_range = []
        for c in range(cent.shape[0]):
            int_range.append(test_img1[int(cent[c,0]),int(cent[c,1])])
        int_array = np.array(int_range) 
        if int_array.size == 0:
            continue
        perc = np.percentile(int_array, 80)
        deep_lesions = np.where(int_array <= perc)[0]
        cent = cent[deep_lesions,:]                                         
        i = 0
        for c in range(cent.shape[0]):
            start_row = (np.amax([cent[c,0] - patch_size//2, 0])).astype(int)
            start_col = (np.amax([cent[c,1] - patch_size//2, 0])).astype(int)
            end_row = (np.amin([cent[c,0] + patch_size//2,volume3d.shape[0]])).astype(int)
            end_col = (np.amin([cent[c,1] + patch_size//2,volume3d.shape[1]])).astype(int)
            crop_patch = volume3d[start_row:end_row, start_col:end_col, st]
            patch_dim = crop_patch.shape
            volume4d[start_row:end_row, start_col:end_col, :, st] = patches5d[i,:patch_dim[0], :patch_dim[1], :]
            i += 1
    volume4d = volume4d.transpose(0,1,3,2)                                         
    return volume4d


def construct_image_from_local_maxima_2Dpatches(filenames,patches4d,patch_size=64):
    volume5d = []
    count_end = 0
    for num_images in range(len(filenames)):
        mc = np.load(filenames[num_images])
        data_sub = mc['data']
        _,coords = data_preprocessing.tight_crop_data(data_sub)
        volume = data_sub[coords[0]:coords[0]+coords[1],coords[2]:coords[2]+coords[3],coords[4]:coords[4]+coords[5]]
        test_img = data_preprocessing.preprocess_data(volume)
        test_img = filters.gaussian_filter(test_img, 0.4)
        #loc_maxima = morphology.h_maxima(test_img,0.2)
        ce = 0
        for st in range(test_img.shape[2]):
            loc_maxima = morphology.h_maxima(test_img[:,:,st],0.2)
            labs,num_labs = morphology.label(loc_maxima,return_num=True)
            props = measure.regionprops(labs)
            cent = np.array([prop.centroid for prop in props])
            int_range = []
            for c in range(cent.shape[0]):
                int_range.append(test_img[int(cent[c,0]),int(cent[c,1])])
            int_array = np.array(int_range) 
            if int_array.size == 0:
                continue
            perc = np.percentile(int_array, 80)
            deep_lesions = np.where(int_array <= perc)[0]
            cent = cent[deep_lesions,:]
            ce += cent.shape[0]
        count_start = count_end
        count_end = count_end + ce 
        patches4d_sub = patches4d[count_start:count_end,:,:,:]
        volume4d = patches2D_from_local_maxima_to_volume(patches4d_sub,volume)
        volume5d.append(volume4d)
    return volume5d


def construct_image_from_2d_patches_from2d_slices(testdata, prob_patches, ca_list, patch_size = 32):
    volume4d = np.zeros([testdata.shape[0],testdata.shape[1],testdata.shape[2],prob_patches.shape[-1]])
    img3d = testdata[:,:,:,0]
    print(volume4d.shape)
    print(img3d.shape[0])
    print(len(ca_list))
    print(ca_list[0].shape)
    c = 0
    for sl in range(img3d.shape[0]):
        img2d = img3d[sl,:,:]
        cent = ca_list[sl]
        for i in range(cent.shape[0]):
            start_row = (np.amax([cent[i,0] - patch_size//2, 0])).astype(int)
            start_col = (np.amax([cent[i,1] - patch_size//2, 0])).astype(int)
            end_row = (np.amin([cent[i,0] + patch_size//2,img2d.shape[0]])).astype(int)
            end_col = (np.amin([cent[i,1] + patch_size//2,img2d.shape[1]])).astype(int)
            crop_patch = img2d[start_row:end_row, start_col:end_col]
            patch_dim = crop_patch.shape
            volume4d[sl,start_row:end_row, start_col:end_col,:] = prob_patches[c,:patch_dim[0], :patch_dim[1],:]
            c += 1
    return volume4d


def construct_image_from_2d_patches_withmargin_from2d_slices(testdata, prob_patches, ca_list, patch_size_org = 32):
    patch_size = patch_size_org - 2
    volume4d = np.zeros([testdata.shape[0],testdata.shape[1],testdata.shape[2],prob_patches.shape[-1]])
    img3d = testdata[:,:,:,0]
    print(volume4d.shape)
    print(img3d.shape[0])
    print(len(ca_list))
    print(ca_list[0].shape)
    c = 0
    for sl in range(img3d.shape[0]):
        img2d = img3d[sl,:,:]
        cent = ca_list[sl]
        for i in range(cent.shape[0]):
            start_row = (np.amax([cent[i,0] - patch_size//2, 0])).astype(int)
            start_col = (np.amax([cent[i,1] - patch_size//2, 0])).astype(int)
            end_row = (np.amin([cent[i,0] + patch_size//2,img2d.shape[0]])).astype(int)
            end_col = (np.amin([cent[i,1] + patch_size//2,img2d.shape[1]])).astype(int)
            crop_patch = img2d[start_row:end_row, start_col:end_col]
            patch_dim = crop_patch.shape
            volume4d[sl,start_row:end_row, start_col:end_col,:] = np.maximum(volume4d[sl,start_row:end_row, start_col:end_col,:],prob_patches[c,2:patch_dim[0]+2, 2:patch_dim[1]+2,:])
            c += 1
    return volume4d

   
def construct_3dvolumes_from_2dslices(volume4d,testpaths):
    volumes = []
    st = 0
    for i in range(len(testpaths)):
        mc = np.load(testpaths[i])
        data = mc['data']
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        data_sub = data[2:2+128,coords[2]:coords[2]+192,coords[4]+12:coords[4]+44]        
        volume3d[2:2+128,coords[2]:coords[2]+192,coords[4]+12:coords[4]+44] = volume4d[st:st+data_sub.shape[2],:,:,1].transpose(1,2,0)
        st = st+data_sub.shape[2]
        volumes.append(volume3d)
    return volumes


def construct_3dvolumes_from_2dslices_p(volume4d,testpaths):
    volumes = []
    st = 0
    for i in range(len(testpaths)):
        mc = np.load(testpaths[i])
        data = mc['data']
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        data_sub = data[2:2+128,coords[2]:coords[2]+192,coords[4]+16:coords[4]+44]
        volume3d[2:2+128,coords[2]:coords[2]+192,coords[4]+16:coords[4]+44] = volume4d[st:st+data_sub.shape[2],:,:,1].transpose(1,2,0)
        st = st+data_sub.shape[2]
        volumes.append(volume3d)
    return volumes


def construct_3dvolumes_from_2dslices_cropped(volume4d,testpaths,vent=0):
    volumes = []
    st = 0
    for i in range(len(testpaths)):
        mc = np.load(testpaths[i])
        data = mc['data']
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        row_cent = coords[1]//2 + coords[0]
        col_cent = coords[3]//2 + coords[2]
        rowstart = np.amax([row_cent-64,0])
        rowend = np.amin([row_cent+64,data.shape[0]])
        colstart = np.amax([col_cent-96,0])
        colend = np.amin([col_cent+96,data.shape[1]])
        if vent==1:
            GM_distance_sub = mc['gmdist']
            ventdistmap_sub = mc['ventdist']
            vent_mask = (ventdistmap_sub == 0).astype(float) * (GM_distance_sub >= 8).astype(float)
            _,coords_vent = data_preprocessing.tight_crop_data(vent_mask)
            stackstart = coords_vent[4]
            stackend = coords_vent[4] + coords_vent[5]
        else:
            stackstart = coords[4]
            stackend = coords[4] + coords[5]
        data_sub = data[rowstart:rowend,colstart:colend,stackstart:stackend]
        print(data_sub.shape)
        print(volume4d.shape)
        print(i, st, st+data_sub.shape[2])
        print(stackend-stackstart)
        required_stacks = volume4d[st:st+data_sub.shape[2],:data_sub.shape[0],:data_sub.shape[1],1].transpose(1,2,0)
        print('Post-processing step')
        print(required_stacks.shape)
        print(stackend-stackstart)
        print(data_sub.shape)
        print(i, st, st+data_sub.shape[2])
        print(volume4d.shape)
        volume3d[rowstart:rowend,colstart:colend,stackstart:stackend] = required_stacks
        st = st+data_sub.shape[2]
        volumes.append(volume3d)
    return volumes


def construct_3dvolumes_from_2dslices_cropped_ox(volume4d,testpaths,vent=0):
    volumes = []
    st = 0
    for i in range(len(testpaths)):
        mc = np.load(testpaths[i])
        data = mc['data']
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        row_cent = coords[1]//2 + coords[0]
        col_cent = coords[3]//2 + coords[2]
        rowstart = np.amax([row_cent-64,0])
        rowend = np.amin([row_cent+64,data.shape[0]])
        colstart = coords[2]
        colend = coords[2] + coords[3]
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        data_sub = data[rowstart:rowend,colstart:colend,stackstart:stackend]
        required_stacks = volume4d[st:st+data_sub.shape[2],:data_sub.shape[0],:data_sub.shape[1],1].transpose(1,2,0)
        print('Post-processing step')
        print(required_stacks.shape)
        print(stackend-stackstart)
        print(data_sub.shape)
        print(i, st, st+data_sub.shape[2])
        print(volume4d.shape)
        volume3d[rowstart:rowend,colstart:colend,stackstart:stackend] = required_stacks
        st = st+data_sub.shape[2]
        volumes.append(volume3d)
    return volumes


def construct_3dvolumes_from_2dslices_cropped_optima(volume4d,testpaths,vent=0):
    volumes = []
    st = 0
    for i in range(len(testpaths)):
        mc = np.load(testpaths[i])
        data = mc['data']
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        row_cent = coords[1]//2 + coords[0]
        col_cent = coords[3]//2 + coords[2]
        rowstart = np.amax([row_cent-96,0])
        rowend = np.amin([row_cent+96,data.shape[0]])
        colstart = np.amax([col_cent-96,0])
        colend = np.amin([col_cent+96,data.shape[1]])
        if vent==1:
            GM_distance_sub = mc['gmdist']
            ventdistmap_sub = mc['ventdist']
            vent_mask = (ventdistmap_sub == 0).astype(float) * (GM_distance_sub >= 8).astype(float)
            _,coords_vent = data_preprocessing.tight_crop_data(vent_mask)
            stackstart = coords_vent[4]
            stackend = coords_vent[4] + coords_vent[5]
        else:
            stackstart = coords[4]
            stackend = coords[4] + coords[5]
        data_sub = data[rowstart:rowend,colstart:colend,stackstart:stackend]
        volume3d[rowstart:rowend,colstart:colend,stackstart:stackend] = volume4d[st:st+data_sub.shape[2],:data_sub.shape[0],:data_sub.shape[1],1].transpose(1,2,0)
        st = st+data_sub.shape[2]
        volumes.append(volume3d)
    return volumes


def resize_to_original_size(probs, testpaths, plane='sagittal'):
    overall_prob = np.array([])
    st = 0
    for i in range(len(testpaths)):
        mc = np.load(testpaths[i])
        data = mc['data']
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        row_cent = coords[1]//2 + coords[0]
        col_cent = coords[3]//2 + coords[2]
        rowstart = np.amax([row_cent-64,0])
        rowend = np.amin([row_cent+64,data.shape[0]])
        colstart = np.amax([col_cent-96,0])
        colend = np.amin([col_cent+96,data.shape[1]])
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        if plane == 'sagittal':
            probs_sub = probs[i*128:(i+1)*128,:,:,:]
            probs_sub_resize = np.zeros([probs_sub.shape[0],probs_sub.shape[1],coords[5]])
            for sli in range(probs_sub.shape[0]):
                probs_sub_resize[sli,:,:] = resize(probs_sub[sli,:,:,1], [probs_sub.shape[1], coords[5]], preserve_range=True)
            prob_specific_sub = probs_sub_resize.transpose(2,0,1)
            overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub 
        elif plane == 'coronal':
            probs_sub = probs[i*192:(i+1)*192,:,:,:]
            probs_sub_resize = np.zeros([probs_sub.shape[0],probs_sub.shape[1],coords[5]])
            for sli in range(probs_sub.shape[0]):
                probs_sub_resize[sli,:,:] = resize(probs_sub[sli,:,:,1], [probs_sub.shape[1], coords[5]], preserve_range=True)
            prob_specific_sub = probs_sub_resize.transpose(2,1,0)
            overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub
    return overall_prob


def resize_to_original_size_opt(probs, testpaths, plane='sagittal'):
    overall_prob = np.array([])
    st = 0
    for i in range(len(testpaths)):
        mc = np.load(testpaths[i])
        data = mc['data']
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        row_cent = coords[1]//2 + coords[0]
        col_cent = coords[3]//2 + coords[2]
        rowstart = np.amax([row_cent-96,0])
        rowend = np.amin([row_cent+96,data.shape[0]])
        colstart = np.amax([col_cent-96,0])
        colend = np.amin([col_cent+96,data.shape[1]])
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        if plane == 'sagittal':
            probs_sub = probs[i*192:(i+1)*192,:,:,:]
            probs_sub_resize = np.zeros([probs_sub.shape[0],probs_sub.shape[1],coords[5]])
            for sli in range(probs_sub.shape[0]):
                probs_sub_resize[sli,:,:] = resize(probs_sub[sli,:,:,1], [probs_sub.shape[1], coords[5]], preserve_range=True)
            prob_specific_sub = probs_sub_resize.transpose(2,0,1)
            overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub 
        elif plane == 'coronal':
            probs_sub = probs[i*192:(i+1)*192,:,:,:]
            probs_sub_resize = np.zeros([probs_sub.shape[0],probs_sub.shape[1],coords[5]])
            for sli in range(probs_sub.shape[0]):
                probs_sub_resize[sli,:,:] = resize(probs_sub[sli,:,:,1], [probs_sub.shape[1], coords[5]], preserve_range=True)
            prob_specific_sub = probs_sub_resize.transpose(2,1,0)
            overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub
    return overall_prob


def resize_to_original_size_ox(probs, testpaths, plane='axial'):
    overall_prob = np.array([])
    st = 0
    for i in range(len(testpaths)):
        mc = np.load(testpaths[i])
        data = mc['data']
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        row_cent = coords[1]//2 + coords[0]
        rowstart = np.amax([row_cent-64,0])
        rowend = np.amin([row_cent+64,data.shape[0]])
        colstart = coords[2]
        colend = coords[2] + coords[3]
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        if plane =='axial':
            probs_sub = probs[st:st+coords[5],:,:,:]
            st = st+coords[5]
            prob_specific_sub = np.zeros([probs_sub.shape[0],probs_sub.shape[1],coords[3]])
            for sli in range(probs_sub.shape[0]):
                prob_specific_sub[sli,:,:] = resize(probs_sub[sli,:,:,1], [probs_sub.shape[1], coords[3]], preserve_range=True)
            print(prob_specific_sub.shape)
            print(probs_sub.shape)
            print(coords[3],coords[5])
            overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub
        elif plane == 'sagittal':
            probs_sub = probs[i*128:(i+1)*128,:,:,:]
            probs_sub_resize = np.zeros([probs_sub.shape[0],coords[3],coords[5]])
            for sli in range(probs_sub.shape[0]):
                probs_sub_resize[sli,:,:] = resize(probs_sub[sli,:,:,1], [coords[3], coords[5]], preserve_range=True)
            prob_specific_sub = probs_sub_resize.transpose(2,0,1)
            overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub 
        elif plane == 'coronal':
            probs_sub = probs[st:st+coords[3],:,:,:]
            st = st+coords[3]
            probs_sub_resize = np.zeros([probs_sub.shape[0],128,coords[5]])
            for sli in range(probs_sub.shape[0]):
                probs_sub_resize[sli,:,:] = resize(probs_sub[sli,:,:,1], [128, coords[5]], preserve_range=True)
            prob_specific_sub = probs_sub_resize.transpose(2,1,0)
            overall_prob = np.concatenate((overall_prob,prob_specific_sub),axis = 0) if overall_prob.size else prob_specific_sub
    return overall_prob


def resize_to_original_size_ukbb_unlab(probs, testpaths, plane='axial'):
    overall_prob = np.array([])
    st = 0
    for i in range(len(testpaths)):
        data = nib.load(
                '/path/to/UKBB/FLAIR_brain.nii.gz').get_data().astype(float)
        _,coords = data_preprocessing.tight_crop_data(data)
        row_cent = coords[1]//2 + coords[0]
        rowstart = np.amax([row_cent-64, 0])
        rowend = np.amin([row_cent+64, data.shape[0]])
        colstart = coords[2]
        colend = coords[2] + coords[3]
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        if plane == 'axial':
            probs_sub = probs[st:st+coords[5], :, :, :]
            st = st+coords[5]
            prob_specific_sub = np.zeros([probs_sub.shape[0], probs_sub.shape[1], coords[3]])
            for sli in range(probs_sub.shape[0]):
                prob_specific_sub[sli, :, :] = resize(probs_sub[sli, :, :, 1], [probs_sub.shape[1], coords[3]],
                                                      preserve_range=True)
            print(prob_specific_sub.shape)
            print(probs_sub.shape)
            print(coords[3], coords[5])
            overall_prob = np.concatenate((overall_prob, prob_specific_sub), axis=0) if overall_prob.size else prob_specific_sub
        elif plane == 'sagittal':
            probs_sub = probs[i*128:(i+1)*128, :, :, :]
            probs_sub_resize = np.zeros([probs_sub.shape[0], coords[3], coords[5]])
            for sli in range(probs_sub.shape[0]):
                probs_sub_resize[sli, :, :] = resize(probs_sub[sli, :, :, 1], [coords[3], coords[5]],
                                                     preserve_range=True)
            prob_specific_sub = probs_sub_resize.transpose(2, 0, 1)
            overall_prob = np.concatenate((overall_prob, prob_specific_sub), axis=0) if overall_prob.size else prob_specific_sub
        elif plane == 'coronal':
            probs_sub = probs[st:st+coords[3], :, :, :]
            st = st+coords[3]
            probs_sub_resize = np.zeros([probs_sub.shape[0], 128, coords[5]])
            for sli in range(probs_sub.shape[0]):
                probs_sub_resize[sli, :, :] = resize(probs_sub[sli, :, :, 1], [128, coords[5]],
                                                     preserve_range=True)
            prob_specific_sub = probs_sub_resize.transpose(2, 1, 0)
            overall_prob = np.concatenate((overall_prob, prob_specific_sub), axis=0) if overall_prob.size else prob_specific_sub
    return overall_prob


def construct_3dvolumes_ukbb(volume4d, testpaths):
    volumes = []
    st = 0
    for i in range(len(testpaths)):
        data = nib.load(
                '/path/to/UKBB_FLAIR_brain.nii.gz').get_data().astype(float)
        volume3d = 0 * data
        _,coords = data_preprocessing.tight_crop_data(data)
        row_cent = coords[1]//2 + coords[0]
        col_cent = coords[3]//2 + coords[2]
        rowstart = np.amax([row_cent - 64, 0])
        rowend = np.amin([row_cent + 64, data.shape[0]])
        colstart = coords[2]
        colend = coords[2] + coords[3]
        stackstart = coords[4]
        stackend = coords[4] + coords[5]
        data_sub = data[rowstart:rowend, colstart:colend, stackstart:stackend]
        required_stacks = volume4d[st:st+data_sub.shape[2], :data_sub.shape[0], :data_sub.shape[1], 1].transpose(1,2,0)
        print('Post-processing step')
        print(required_stacks.shape)
        print(stackend-stackstart)
        print(data_sub.shape)
        print(i, st, st+data_sub.shape[2])
        print(volume4d.shape)
        volume3d[rowstart:rowend, colstart:colend, stackstart:stackend] = required_stacks
        st = st+data_sub.shape[2]
        volumes.append(volume3d)
    return volumes


