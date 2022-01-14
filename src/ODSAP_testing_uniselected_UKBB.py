#!/usr/bin/env python
#   Copyright (C) 2016 University of Oxford 
#   SHBASECOPYRIGHT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from loss_functions_torch import *
import glob
import nibabel as nib
from data_preparation_pytorch_dann import *
from data_postprocessing_pytorch_dann import *
from DANN_truenet_model_encoder_end_unnamed import TrUENet, Domain_Predictor
from DANN_truenet_model_encoder_end import TrUENet as TrUENet1
from DANN_truenet_model_encoder_end import Domain_Predictor as Domain_Predictor1


def append_data(testdata):
    for d in range(len(testdata)):
        da = testdata[d]
        if len(da.shape) == 4:
            extra = np.zeros([8, da.shape[1], da.shape[2], da.shape[3]])
        else:
            extra = np.zeros([8, da.shape[1], da.shape[2]])
        da = np.concatenate([da, extra], axis=0)
        testdata[d] = da
    return testdata


def dice_coeff(inp, tar):
    smooth = 1.
    pred_vect = inp.contiguous().view(-1)
    target_vect = tar.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice


def twolists(list1, list2):
    newlist = []
    a1 = len(list1)
    a2 = len(list2)

    for i in range(max(a1, a2)):
        if i < a1:
            newlist.append(list1[i])
        if i < a2:
            newlist.append(list2[i])

    return newlist


def eval_truenet(testdata, models, batch_size, device, test=0, mode='axial'):
    dir_checkpoint = '/path/to/model/checkpoints/'
    [truenet, domain_predictor] = models
    truenet.eval()
    for keys in truenet.state_dict().keys():
        print(keys)
    truenet.load_state_dict(torch.load(dir_checkpoint + 'CP_semiDANN_omni_pretrain_epoch10_' + mode + '.pth'))
    domain_predictor.eval()
    if test:
        testdata = append_data(testdata)
    nsteps = max(testdata[0].shape[0] // batch_size, 1)
    prob_array = np.array([])
    gen_next_test_batch = batch_generator(testdata, batch_size, shuffle=False)
    for i in range(nsteps):
        Xv, _, _ = next(gen_next_test_batch)
        Xv = Xv.transpose(0, 3, 1, 2)
        Xv = torch.from_numpy(Xv)
        Xv = Xv.to(device=device, dtype=torch.float32)
        val_pred, bottleneck = truenet(Xv)
        del bottleneck
        softmax = nn.Softmax()
        probs = softmax(val_pred)
        probs1 = probs.cpu()
        probs_np = probs1.detach().numpy()
        
        prob_array = np.concatenate((prob_array, probs_np), axis=0) if prob_array.size else probs_np
    prob_array = prob_array.transpose(0, 2, 3, 1)
    return prob_array


ukbb_names = [21567098, 21968981, 22288821, 22405083, 22648553, 22962221, 23258920, 23286836,
              23734335, 23965250, 24374960, 24387742]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = [str(ukbb_name) for ukbb_name in ukbb_names]
output_folder = '/path/to/Pretrained_results/'
batch_size = 1

for i, name in enumerate(data_path):
    test_names = [name]
    data_hdr = nib.load(
            'path/to/UKBB_FLAIR_brain.nii.gz').header
    prob_image_3planes = []

    models_axial = [TrUENet(n_channels=2, n_classes=2, batch_size=8, init_channels=64, plane='axial'),
                    Domain_Predictor(n_domains=2,  plane='axial')]
    models_sagittal = [TrUENet(n_channels=2, n_classes=2, batch_size=8, init_channels=64, plane='sagittal'),
                       Domain_Predictor(n_domains=2,  plane='sagittal')]
    models_coronal = [TrUENet1(n_channels=2, n_classes=2, batch_size=8, init_channels=64, plane='coronal'),
                      Domain_Predictor1(n_domains=2,  plane='coronal')]

    [truenet_axial, domain_predictor_axial] = models_axial
    [truenet_sagittal, domain_predictor_sagittal] = models_sagittal
    [truenet_coronal, domain_predictor_coronal] = models_coronal

    truenet_axial.to(device=device)
    truenet_sagittal.to(device=device)
    truenet_coronal.to(device=device)
    truenet_axial = nn.DataParallel(truenet_axial)
    truenet_sagittal = nn.DataParallel(truenet_sagittal)
    truenet_coronal = nn.DataParallel(truenet_coronal)

    domain_predictor_axial.to(device=device)
    domain_predictor_sagittal.to(device=device)
    domain_predictor_coronal.to(device=device)
    domain_predictor_axial = nn.DataParallel(domain_predictor_axial)
    domain_predictor_sagittal = nn.DataParallel(domain_predictor_sagittal)
    domain_predictor_coronal = nn.DataParallel(domain_predictor_coronal)

    model_axial = [truenet_axial, domain_predictor_axial]
    model_sagittal = [truenet_sagittal, domain_predictor_sagittal]
    model_coronal = [truenet_coronal, domain_predictor_coronal]

    brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ukb_unlab(test_names,
                                                                                                   plane='axial')
    testdata = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                   ventdistmap, plane='axial', test=1)
    prob_vals1_axial = 0 * testdata[0]
    prob_array_axial = eval_truenet(testdata, model_axial, batch_size, device, test=1, mode='axial')

    print('Testdata Axial dimensions............................................')
    print(testdata[0].shape)  
    print('Probability dimensions............................................')
    print(prob_array_axial.shape)
    
    prob_vals1_axial = resize_to_original_size_ukbb_unlab(prob_vals1_axial, test_names, plane='axial')
    prob_image_3planes.append(prob_vals1_axial)

    brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ukb_unlab(test_names,
                                                                                                   plane='sagittal')
    testdata = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                   ventdistmap, plane='sagittal', test=1)
    prob_array_sagittal = eval_truenet(testdata, model_sagittal, batch_size, device, test=1, mode='sagittal')

    print('Testdata Sagittal dimensions............................................')
    print(testdata[0].shape)  
    print('Probability dimensions............................................')
    print(prob_array_sagittal.shape)

    prob_vals1_sagittal = 0*testdata[0]
    prob_vals1_sagittal[:prob_array_sagittal.shape[0],:prob_array_sagittal.shape[1],:prob_array_sagittal.shape[2],:] = prob_array_sagittal
    prob_vals1_sagittal = resize_to_original_size_ukbb_unlab(prob_vals1_sagittal, test_names, plane='sagittal')
    prob_image_3planes.append(prob_vals1_sagittal)

    brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ukb_unlab(test_names,
                                                                                                   plane='coronal')
    testdata = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                   ventdistmap, plane='coronal', test=1)
    prob_array_coronal = eval_truenet(testdata, model_coronal, batch_size, device, test=1, mode='coronal')

    print('Testdata Coronal dimensions............................................')
    print(testdata[0].shape)  
    print('Probability dimensions............................................')
    print(prob_array_coronal.shape)

    prob_vals1_coronal = 0*testdata[0]
    prob_vals1_coronal[:prob_array_coronal.shape[0], :prob_array_coronal.shape[1], :prob_array_coronal.shape[2], :] = prob_array_coronal
    prob_vals1_coronal = resize_to_original_size_ukbb_unlab(prob_vals1_coronal, test_names, plane='coronal')
    prob_image_3planes.append(prob_vals1_coronal)

    print('Aggregates probability dimensions............................................')
    print(prob_image_3planes[0].shape)
    print(prob_image_3planes[1].shape)
    print(prob_image_3planes[2].shape)

    prob_image_3planes = np.array(prob_image_3planes)
    prob_mean = np.mean(prob_image_3planes, axis=0)
    prob_mean = np.tile(prob_mean, (1, 1, 1, 1))
    prob_mean = prob_mean.transpose(1, 2, 3, 0)
    prob_mean = np.concatenate((prob_mean, prob_mean), axis=-1)
    volumes_m = construct_3dvolumes_ukbb(prob_mean, test_names)
    del prob_mean

    probobj = nib.nifti1.Nifti1Image(volumes_m[0], None, header=data_hdr)
    nib.save(probobj, output_folder + 'Probsmaps_pretrained_OSDAP_' + name + '.nii.gz')
