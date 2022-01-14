#!/usr/bin/env python
#   Copyright (C) 2016 University of Oxford
#   SHBASECOPYRIGHT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nibabel as nib
from utils import *
import augmentations_distmaps_t1
from skimage.transform import resize
from scipy import ndimage
import data_preprocessing


def create_data_array_from_loaded_data_ukb(names, plane='axial'):
    data = np.array([])
    data_t1 = np.array([])
    brains = np.array([])
    GM_distance = np.array([])
    ventdistmap = np.array([])

    if plane == 'axial':
        for i in range(len(names)):
            try:
                data_sub1, brain_sub1, data_t1_sub1, GM_distance_sub1, ventdistmap_sub1 = load_and_crop_2d_data_ukb(names[i])
            except:
                continue
            data_sub1 = data_sub1.transpose(2, 0, 1)
            brain_sub1 = brain_sub1.transpose(2, 0, 1)
            data_t1_sub1 = data_t1_sub1.transpose(2, 0, 1)
            GM_distance_sub1 = GM_distance_sub1.transpose(2, 0, 1)
            ventdistmap_sub1 = ventdistmap_sub1.transpose(2, 0, 1)
            data = np.concatenate(
                (data, resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 192], preserve_range=True)),
                axis=0) if data.size else resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 192],
                                                 preserve_range=True)
            brains = np.concatenate(
                (brains, resize(brain_sub1, [brain_sub1.shape[0], brain_sub1.shape[1], 192], preserve_range=True)),
                axis=0) if brains.size else resize(brain_sub1, [brain_sub1.shape[0], brain_sub1.shape[1], 192],
                                                   preserve_range=True)
            data_t1 = np.concatenate((data_t1, resize(data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 192],
                                                      preserve_range=True)), axis=0) if data_t1.size else resize(
                data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 192], preserve_range=True)
            GM_distance = np.concatenate((GM_distance, resize(GM_distance_sub1,
                                                              [GM_distance_sub1.shape[0], GM_distance_sub1.shape[1],
                                                               192], preserve_range=True)),
                                         axis=0) if GM_distance.size else resize(GM_distance_sub1,
                                                                                 [GM_distance_sub1.shape[0],
                                                                                  GM_distance_sub1.shape[1], 192],
                                                                                 preserve_range=True)
            ventdistmap = np.concatenate((ventdistmap, resize(ventdistmap_sub1,
                                                              [ventdistmap_sub1.shape[0], ventdistmap_sub1.shape[1],
                                                               192], preserve_range=True)),
                                         axis=0) if ventdistmap.size else resize(ventdistmap_sub1,
                                                                                 [ventdistmap_sub1.shape[0],
                                                                                  ventdistmap_sub1.shape[1], 192],
                                                                                 preserve_range=True)
    elif plane == 'sagittal':
        for i in range(len(names)):
            try:
                data_sub1, brain_sub1, data_t1_sub1, GM_distance_sub1, ventdistmap_sub1 = load_and_crop_2d_data_ukb(
                names[i])
            except:
                continue
            data = np.concatenate((data, resize(data_sub1, [data_sub1.shape[0], 192, 120], preserve_range=True)),
                                  axis=0) if data.size else resize(data_sub1, [data_sub1.shape[0], 192, 120],
                                                                   preserve_range=True)
            brains = np.concatenate((brains, resize(brain_sub1, [brain_sub1.shape[0], 192, 120], preserve_range=True)),
                                    axis=0) if brains.size else resize(brain_sub1, [brain_sub1.shape[0], 192, 120],
                                                                       preserve_range=True)
            data_t1 = np.concatenate(
                (data_t1, resize(data_t1_sub1, [data_t1_sub1.shape[0], 192, 120], preserve_range=True)),
                axis=0) if data_t1.size else resize(data_t1_sub1, [data_t1_sub1.shape[0], 192, 120],
                                                    preserve_range=True)
            GM_distance = np.concatenate(
                (GM_distance, resize(GM_distance_sub1, [GM_distance_sub1.shape[0], 192, 120], preserve_range=True)),
                axis=0) if GM_distance.size else resize(GM_distance_sub1, [GM_distance_sub1.shape[0], 192, 120],
                                                        preserve_range=True)
            ventdistmap = np.concatenate(
                (ventdistmap, resize(ventdistmap_sub1, [ventdistmap_sub1.shape[0], 192, 120], preserve_range=True)),
                axis=0) if ventdistmap.size else resize(ventdistmap_sub1, [ventdistmap_sub1.shape[0], 192, 120],
                                                        preserve_range=True)
    elif plane == 'coronal':
        for i in range(len(names)):
            try:
                data_sub1, brain_sub1, data_t1_sub1, GM_distance_sub1, ventdistmap_sub1 = load_and_crop_2d_data_ukb(
                    names[i])
            except:
                continue
            data_sub1 = data_sub1.transpose(1, 0, 2)
            brain_sub1 = brain_sub1.transpose(1, 0, 2)
            data_t1_sub1 = data_t1_sub1.transpose(1, 0, 2)
            GM_distance_sub1 = GM_distance_sub1.transpose(1, 0, 2)
            ventdistmap_sub1 = ventdistmap_sub1.transpose(1, 0, 2)
            data = np.concatenate(
                (data, resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 80], preserve_range=True)),
                axis=0) if data.size else resize(data_sub1, [data_sub1.shape[0], data_sub1.shape[1], 80],
                                                 preserve_range=True)
            brains = np.concatenate(
                (brains, resize(brain_sub1, [brain_sub1.shape[0], brain_sub1.shape[1], 80], preserve_range=True)),
                axis=0) if brains.size else resize(brain_sub1, [brain_sub1.shape[0], brain_sub1.shape[1], 80],
                                                   preserve_range=True)
            data_t1 = np.concatenate((data_t1, resize(data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 80],
                                                      preserve_range=True)), axis=0) if data_t1.size else resize(
                data_t1_sub1, [data_t1_sub1.shape[0], data_t1_sub1.shape[1], 80], preserve_range=True)
            GM_distance = np.concatenate((GM_distance, resize(GM_distance_sub1,
                                                              [GM_distance_sub1.shape[0], GM_distance_sub1.shape[1],
                                                               80], preserve_range=True)),
                                         axis=0) if GM_distance.size else resize(GM_distance_sub1,
                                                                                 [GM_distance_sub1.shape[0],
                                                                                  GM_distance_sub1.shape[1], 80],
                                                                                 preserve_range=True)
            ventdistmap = np.concatenate((ventdistmap, resize(ventdistmap_sub1,
                                                              [ventdistmap_sub1.shape[0], ventdistmap_sub1.shape[1],
                                                               80], preserve_range=True)),
                                         axis=0) if ventdistmap.size else resize(ventdistmap_sub1,
                                                                                 [ventdistmap_sub1.shape[0],
                                                                                  ventdistmap_sub1.shape[1], 80],
                                                                                 preserve_range=True)

    data = np.tile(data, (1, 1, 1, 1))
    data = data.transpose(1, 2, 3, 0)
    data_t1 = np.tile(data_t1, (1, 1, 1, 1))
    data_t1 = data_t1.transpose(1, 2, 3, 0)
    return brains, data, data_t1, GM_distance, ventdistmap


def load_and_crop_2d_data_ukb(data_path):
    data_sub_org = nib.load(
            '/path/to/UKBB_FLAIR_brain.nii.gz').get_data().astype(float)
    data_t1_sub_org = nib.load(
        '/path/to/UKBB_T1_brain.nii.gz').get_data().astype(float)
    ventmask = nib.load(
            '/path/to/UKBB_ventmask.nii.gz').get_data().astype(float)
    bianca_mask = nib.load(
            'path/to/UKBB_bianca_mask.nii.gz').get_data().astype(float)
    brain_mask = nib.load(
            '/path/to/UKBB_brain_mask.nii.gz').get_data().astype(float)
    labels_sub = nib.load('/path/to/UKBB_lesion_mask.nii.gz').get_data().astype(float)

    non_gmmask = ventmask + bianca_mask
    GM_distance_sub = (ndimage.distance_transform_edt(non_gmmask)) * bianca_mask
    ventdistmap_sub = (ndimage.distance_transform_edt(brain_mask - ventmask)) * bianca_mask

    _, coords = data_preprocessing.tight_crop_data(data_sub_org)
    row_cent = coords[1] // 2 + coords[0]
    rowstart = np.amax([row_cent - 64, 0])
    rowend = np.amin([row_cent + 64, data_sub_org.shape[0]])
    colstart = coords[2]
    colend = coords[2] + coords[3]
    stackstart = coords[4]
    stackend = coords[4] + coords[5]
    data_sub1 = np.zeros([128, coords[3], coords[5]])
    labels_sub1 = np.zeros([128, coords[3], coords[5]])
    brain_sub1 = np.zeros([128, coords[3], coords[5]])
    data_t1_sub1 = np.zeros([128, coords[3], coords[5]])
    GM_distance_sub1 = np.zeros([128, coords[3], coords[5]])
    ventdistmap_sub1 = np.zeros([128, coords[3], coords[5]])
    data_sub_piece = data_preprocessing.preprocess_data_gauss(
        data_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
    brain_sub_piece = brain_mask[rowstart:rowend, colstart:colend, stackstart:stackend]
    data_t1_sub_piece = data_preprocessing.preprocess_data_gauss(
        data_t1_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
    labels_sub_piece = labels_sub[rowstart:rowend, colstart:colend, stackstart:stackend]
    GM_distance_sub_piece = GM_distance_sub[rowstart:rowend, colstart:colend, stackstart:stackend]
    ventdistmap_sub_piece = ventdistmap_sub[rowstart:rowend, colstart:colend, stackstart:stackend]
    data_sub1[:data_sub_piece.shape[0], :data_sub_piece.shape[1], :data_sub_piece.shape[2]] = data_sub_piece
    brain_sub1[:brain_sub_piece.shape[0], :brain_sub_piece.shape[1], :brain_sub_piece.shape[2]] = brain_sub_piece
    labels_sub1[:labels_sub_piece.shape[0], :labels_sub_piece.shape[1], :labels_sub_piece.shape[2]] = labels_sub_piece
    data_t1_sub1[:data_t1_sub_piece.shape[0], :data_t1_sub_piece.shape[1], :data_t1_sub_piece.shape[2]] = data_t1_sub_piece
    GM_distance_sub1[:GM_distance_sub_piece.shape[0], :GM_distance_sub_piece.shape[1], :GM_distance_sub_piece.shape[2]] = GM_distance_sub_piece
    ventdistmap_sub1[:ventdistmap_sub_piece.shape[0], :ventdistmap_sub_piece.shape[1], :ventdistmap_sub_piece.shape[2]] = ventdistmap_sub_piece
    return data_sub1, brain_sub1, labels_sub1, data_t1_sub1, GM_distance_sub1, ventdistmap_sub1


def get_slices_from_data_with_aug(data, data_t1, labels, GM_distance, ventdistmap, plane='axial', test=0):
    gm_distance = 10 * GM_distance ** 0.33
    ventdistmap = 6 * (ventdistmap ** .5)

    if plane == 'sagittal':
        aug_factor = 2
    elif plane == 'coronal':
        aug_factor = 2
    elif plane == 'axial':
        aug_factor = 3

    if test == 0:
        data, data_t1, labels, GM_distance, ventdistmap = perform_augmentation(data, data_t1, labels,
                                                                               GM_distance, ventdistmap,
                                                                               aug_factor)
    tr_data = np.concatenate((data, data_t1), axis=-1)

    data2d = [tr_data, labels, GM_distance, ventdistmap]

    return data2d


def perform_augmentation(otr, otr_t1, otr_labs, otr_gmdist, otr_ventdistmap, af=20):
    augmented_img_list = []
    augmented_img_t1_list = []
    augmented_mseg_list = []
    augmented_gmdist_list = []
    augmented_ventdist_list = []
    for i in range(0,af):
        for id in range(otr.shape[0]):
            image = otr[id,:,:,0]
            image_t1 = otr_t1[id,:,:,0]
            manmask = otr_labs[id,:,:]
            gmdist = otr_gmdist[id,:,:]
            ventdist = otr_ventdistmap[id,:,:]
            augmented_img, augmented_img_t1, augmented_manseg, augmented_gmdist, augmented_ventdist = augmentations_distmaps_t1.augment(image,image_t1,manmask,gmdist,ventdist)
            augmented_img_list.append(augmented_img)
            augmented_img_t1_list.append(augmented_img_t1)
            augmented_mseg_list.append(augmented_manseg)
            augmented_gmdist_list.append(augmented_gmdist)
            augmented_ventdist_list.append(augmented_ventdist)
    augmented_img = np.array(augmented_img_list)
    augmented_img_t1 = np.array(augmented_img_t1_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_gmdist = np.array(augmented_gmdist_list)
    augmented_ventdist = np.array(augmented_ventdist_list)
    augmented_img = np.reshape(augmented_img,[-1,otr.shape[1],otr.shape[2]])
    augmented_img_t1 = np.reshape(augmented_img_t1,[-1,otr.shape[1],otr.shape[2]])
    augmented_mseg = np.reshape(augmented_mseg,[-1,otr.shape[1],otr.shape[2]])
    augmented_gmdist = np.reshape(augmented_gmdist,[-1,otr.shape[1],otr.shape[2]])
    augmented_ventdist = np.reshape(augmented_ventdist,[-1,otr.shape[1],otr.shape[2]])
    augmented_img = np.tile(augmented_img,(1,1,1,1))
    augmented_imgs = augmented_img.transpose(1,2,3,0)
    augmented_img_t1 = np.tile(augmented_img_t1,(1,1,1,1))
    augmented_imgs_t1 = augmented_img_t1.transpose(1,2,3,0)
    otr_aug = np.concatenate((otr,augmented_imgs),axis=0)
    otr_aug_t1 = np.concatenate((otr_t1,augmented_imgs_t1),axis=0)
    otr_labs = np.concatenate((otr_labs,augmented_mseg),axis = 0)
    otr_gmdist = np.concatenate((otr_gmdist,augmented_gmdist),axis = 0)
    otr_ventdistmap = np.concatenate((otr_ventdistmap,augmented_ventdist),axis = 0)
    return otr_aug, otr_aug_t1, otr_labs, otr_gmdist, otr_ventdistmap