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
# from tqdm import tqdm
import os
from OSDAP_utils import *
import pandas as pd
from scipy.ndimage import filters
import math
from loss_functions_torch import *
import glob
from DANN_truenet_model_encoder_end import TrUENet, Domain_Predictor
from OSDAP_train_utils_ukbb import train_semidann_omnisup


#TODO: define training arguments
args = Args()
args.channels_first = True
args.epochs = 100                 # Total number of epochs
args.pretrain_epochs = 10
args.batch_size = 8
args.diff_model_flag = False
args.alpha = 1
args.beta = 20                  # Loss weights probably change this one if not stable
args.patience = 20              # How many epochs to wait for improvement
args.learning_rate = 1e-3       # Main task lr -> see below for more learning rates
args.epoch_reached = 1          # Epoch to start training from
args.epochs_stage_1 = 2
args.batch_factor_source = 9  # determines how many images are loaded for training at an iteration
args.batch_factor_target = 9  # determines how many images are loaded for training at an iteration
args.batch_factor_source_val = 4
args.batch_factor_target_val = 5
args.target_label_prop = 0.200


dir_checkpoint = '/path/to/model_checkpoints/'
pretrained_modelpath = 'path/to/checkpoints_source_domain/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('age_predictions.csv')
ukbb_ids = df['Subj ID'].values
ukbb_ids = list(ukbb_ids)
ukbb_ids = [str(ukbb_id) for ukbb_id in ukbb_ids]
true_age = df['True Age'].values
delta = df['delta_abs'].values
younger = true_age < 63
elder = true_age > 63
lowerror = delta <= 3.14
higherror = delta > 3.14
young_lowerror = np.where(younger.astype(float) * lowerror.astype(float))[0]
young_higherror = np.where(younger.astype(float) * higherror.astype(float))[0]
elder_lowerror = np.where(elder.astype(float) * lowerror.astype(float))[0]
elder_higherror = np.where(elder.astype(float) * higherror.astype(float))[0]

# TODO: Labelled all groups (neurodegeneration and healthy) uniformly
num_in_each_grp = int((len(ukbb_ids) * args.target_label_prop) // 4)
ylerror = np.random.choice(young_lowerror, size=num_in_each_grp, replace=False)
ukbb_yl_ids = [ukbb_ids[ind] for ind in list(ylerror)]
non_yle_inds = np.setdiff1d(young_lowerror, np.array(ylerror))
ukbb_non_yl_ids = [ukbb_ids[ind] for ind in list(non_yle_inds)]

yherror = np.random.choice(young_higherror, size=num_in_each_grp, replace=False)
ukbb_yh_ids = [ukbb_ids[ind] for ind in list(yherror)]
non_yhe_inds = np.setdiff1d(young_higherror, np.array(yherror))
ukbb_non_yh_ids = [ukbb_ids[ind] for ind in list(non_yhe_inds)]

elerror = np.random.choice(elder_lowerror, size=num_in_each_grp, replace=False)
ukbb_el_ids = [ukbb_ids[ind] for ind in list(elerror)]
non_ele_inds = np.setdiff1d(elder_lowerror, np.array(elerror))
ukbb_non_el_ids = [ukbb_ids[ind] for ind in list(non_ele_inds)]

eherror = np.random.choice(elder_higherror, size=num_in_each_grp, replace=False)
ukbb_eh_ids = [ukbb_ids[ind] for ind in list(eherror)]
non_ehe_inds = np.setdiff1d(elder_higherror, np.array(eherror))
ukbb_non_eh_ids = [ukbb_ids[ind] for ind in list(non_ehe_inds)]

labeled_target = ukbb_yl_ids + ukbb_yh_ids + ukbb_el_ids + ukbb_eh_ids
unlabeled_target = ukbb_non_yl_ids + ukbb_non_yh_ids + ukbb_non_el_ids + ukbb_non_eh_ids

# # TODO: Labelled neurodegeneration only
# num_in_each_grp = int((len(ukbb_ids) * args.target_label_prop) // 2)
# ukbb_yl_ids = [ukbb_ids[ind] for ind in list(young_lowerror)]
#
# yherror = np.random.choice(young_higherror, size=num_in_each_grp, replace=False)
# ukbb_yh_ids = [ukbb_ids[ind] for ind in list(yherror)]
# non_yhe_inds = np.setdiff1d(young_higherror, np.array(yherror))
# ukbb_non_yh_ids = [ukbb_ids[ind] for ind in list(non_yhe_inds)]
#
# ukbb_el_ids = [ukbb_ids[ind] for ind in list(elder_lowerror)]
#
# eherror = np.random.choice(elder_higherror, size=num_in_each_grp, replace=False)
# ukbb_eh_ids = [ukbb_ids[ind] for ind in list(eherror)]
# non_ehe_inds = np.setdiff1d(elder_higherror, np.array(eherror))
# ukbb_non_eh_ids = [ukbb_ids[ind] for ind in list(non_ehe_inds)]
#
# labeled_target = ukbb_yh_ids + ukbb_eh_ids
# unlabeled_target = ukbb_yl_ids + ukbb_non_yh_ids + ukbb_el_ids + ukbb_non_eh_ids


# Define training and testing sets
train_names_source = mwsc_ge3t_path[:18] + mwsc_sing_path[:18] + mwsc_utr_path[:18]
val_names_source = mwsc_ge3t_path[18:] + mwsc_sing_path[18:] + mwsc_utr_path[18:]
train_names_target = ukbb_ids[:900]
val_names_target = ukbb_ids[900:]

train_names_lab_target = labeled_target[:int((len(labeled_target) * 0.90) // 1)]
train_names_unlab_target = unlabeled_target[:int((len(unlabeled_target) * 0.90) // 1)]
val_names_lab_target = labeled_target[int((len(labeled_target) * 0.90) // 1):]
val_names_unlab_target = unlabeled_target[int((len(unlabeled_target) * 0.90) // 1):]
print('Input files names listing...................................')
print(train_names_lab_target)
print(val_names_lab_target)
print(train_names_unlab_target)
print(val_names_unlab_target)

models_axial = [TrUENet(n_channels=2, n_classes=2, batch_size=args.batch_size, init_channels=64, plane='axial'),
                Domain_Predictor(n_domains=2, plane='axial')]
models_sagittal = [TrUENet(n_channels=2, n_classes=2, batch_size=args.batch_size, init_channels=64, plane='sagittal'),
                   Domain_Predictor(n_domains=2, plane='sagittal')]
models_coronal = [TrUENet(n_channels=2, n_classes=2, batch_size=args.batch_size, init_channels=64, plane='coronal'),
                  Domain_Predictor(n_domains=2,  plane='coronal')]

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

model_path = os.path.join(pretrained_modelpath, 'CP_source_pretrained_model_epoch41_axial.pth')
truenet_axial = loading_model(model_path, truenet_axial)
model_path = os.path.join(pretrained_modelpath, 'CP_source_pretrained_model_epoch41_sagittal.pth')
truenet_sagittal = loading_model(model_path, truenet_sagittal)
model_path = os.path.join(pretrained_modelpath, 'CP_source_pretrained_model_epoch41_coronal.pth')
truenet_coronal = loading_model(model_path, truenet_coronal)

models_axial = [truenet_axial, domain_predictor_axial]
models_sagittal = [truenet_sagittal, domain_predictor_sagittal]
models_coronal = [truenet_coronal, domain_predictor_coronal]

sub_names = [train_names_source, train_names_lab_target, train_names_unlab_target,
             val_names_source, val_names_lab_target, val_names_unlab_target]

train_semidann_omnisup(args, sub_names, dir_checkpoint,
                       models_axial, args.batch_size, args.epochs, device, mode='axial')
# train_semidann_omnisup(args, sub_names, dir_checkpoint,
#                        models_sagittal, args.batch_size, args.epochs, device, mode='sagittal')
# train_semidann_omnisup(args, sub_names, dir_checkpoint,
#                        models_coronal, args.batch_size, args.epochs, device, mode='coronal')






