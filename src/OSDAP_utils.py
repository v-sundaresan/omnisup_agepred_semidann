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
from sklearn.utils import shuffle
import model_layers_functions_torch
from loss_functions_torch import *
from data_preparation_pytorch_dann import *


class Args:
    # Store lots of the parameters that we might need to train the model
    def __init__(self):
        self.batch_size = 8
        self.batch_factor_source = 2  # determines how many images are loaded for training at an iteration
        self.batch_factor_target = 2  # determines how many images are loaded for training at an iteration
        self.batch_factor_target_val = 1
        self.batch_factor_source_val = 2
        self.target_label_prop = 0.250
        self.log_interval = 10
        self.learning_rate = 1e-4
        self.learning_rate_dom = 1e-3
        self.eps = 1e-4
        self.momentum = 0.9
        self.epochs = 2
        self.pretrain_epochs = 1
        self.train_val_prop = 0.9
        self.patience = 5
        self.channels_first = True
        self.diff_model_flag = False
        self.alpha = 1
        self.epoch_reached = 1
        self.epochs_stage_1 = 60
        self.beta = 20


class EarlyStoppingModelCheckpointing:
    '''
    Early stopping stops the training if the validation loss doesnt improve after a given patience
    '''

    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, val_dice, best_val_dice, model, epoch, optimizer, scheduler, loss,
                 weights=True, checkpoint=True, save_condition='best', model_path=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_dice, best_val_dice, model, epoch, optimizer, scheduler, loss,
                                 weights, checkpoint, save_condition, model_path)
        elif score < self.best_score:  # Here is the criteria for activation of early stopping counter.
            self.counter += 1
            print('Early Stopping Counter: ', self.counter, '/', self.patience)
            if self.counter >= self.patience:  # When the counter reaches pateince, ES flag is activated.
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_dice, best_val_dice, model, epoch, optimizer, scheduler, loss,
                                 weights, checkpoint, save_condition, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, best_val_acc, model, epoch, optimizer, scheduler, loss,
                        tr_prms, weights, checkpoint, save_condition, PATH):
        # Saving checkpoints
        if checkpoint:
            # Saves the model when the validation loss decreases
            if self.verbose:
                print('Validation loss increased; Saving model ...')
            if weights:
                if save_condition == 'best':
                    save_path = os.path.join(PATH, 'Truenet_model_weights_bestdice.pth')
                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), save_path)
                elif save_condition == 'everyN':
                    N = 10
                    if (epoch % N) == 0:
                        save_path = os.path.join(PATH,
                                                 'Truenet_model_weights_epoch' + str(epoch) + '.pth')
                        torch.save(model.state_dict(), save_path)
                elif save_condition == 'last':
                    save_path = os.path.join(PATH, 'Truenet_model_weights_beforeES.pth')
                    torch.save(model.state_dict(), save_path)
                else:
                    raise ValueError("Invalid saving condition provided! Valid options: best, everyN, last")
            else:
                if save_condition == 'best':
                    save_path = os.path.join(PATH, 'Truenet_model_bestdice.pth')
                    if val_acc > best_val_acc:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_stat_dict': scheduler.state_dict(),
                            'loss': loss
                        }, save_path)
                elif save_condition == 'everyN':
                    N = 10
                    if (epoch % N) == 0:
                        save_path = os.path.join(PATH, 'Truenet_model_epoch' + str(epoch) + '.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_stat_dict': scheduler.state_dict(),
                            'loss': loss
                        }, save_path)
                elif save_condition == 'last':
                    save_path = os.path.join(PATH, 'Truenet_model_beforeES.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_stat_dict': scheduler.state_dict(),
                        'loss': loss
                    }, save_path)
                else:
                    raise ValueError("Invalid saving condition provided! Valid options: best, everyN, last")
        else:
            if self.verbose:
                print('Validation loss increased; Exiting without saving the model ...')


def loading_model(model_name, model, mode='weights'):
    if mode == 'weights':
        try:
            axial_state_dict = torch.load(model_name)
        except:
            axial_state_dict = torch.load(model_name, map_location='cpu')
    else:
        try:
            ckpt = torch.load(model_name)
        except:
            ckpt = torch.load(model_name, map_location='cpu')
        axial_state_dict = ckpt['model_state_dict']

    new_axial_state_dict = OrderedDict()
    for key, value in axial_state_dict.items():
        if 'module.' in key[:7]:
            name = key  # remove `module.`
        else:
            name = 'module.' + key
        new_axial_state_dict[name] = value
    model.load_state_dict(new_axial_state_dict)
    return model


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


def append_source_target_data(source, target):
    for d in range(len(source)):
        da = source[d]
        dat = target[d]
        target_size = dat.shape[0] // 4
        da = np.concatenate([da, dat[:target_size, ...]], axis=0)
        source[d] = da
    return source


def dice_coeff(inp, tar):
    smooth = 1.
    pred_vect = inp.contiguous().view(-1)
    target_vect = tar.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice


def get_training_validation_batch_generators(args, trainnames, valnames, ep, iter, mode, dataset='mwsc'):
    [train_names_source, train_names_lab_target, train_names_unlab_target] = trainnames
    [val_names_source, val_names_lab_target, val_names_unlab_target] = valnames
    trainnames_source = train_names_source[iter * args.batch_factor_source:(iter + 1) * args.batch_factor_source]
    train_labtarget_idx = np.random.randint(0, len(train_names_lab_target), size=(1, args.batch_factor_target))
    train_labtarget_idx = list(train_labtarget_idx[0])
    train_unlabtarget_idx = np.random.randint(0, len(train_names_unlab_target), size=(1, args.batch_factor_target))
    train_unlabtarget_idx = list(train_unlabtarget_idx[0])
    trainnames_labtarget = [train_names_lab_target[idx] for idx in train_labtarget_idx]
    trainnames_unlabtarget = [train_names_unlab_target[idx] for idx in train_unlabtarget_idx]

    val_source_idx = np.random.randint(0, len(val_names_source), size=(1, args.batch_factor_source_val))
    val_source_idx = list(val_source_idx[0])
    valnames_source = [val_names_source[idx] for idx in val_source_idx]
    val_labtarget_idx = np.random.randint(0, len(val_names_lab_target), size=(1, args.batch_factor_target_val))
    val_labtarget_idx = list(val_labtarget_idx[0])
    val_unlabtarget_idx = np.random.randint(0, len(val_names_unlab_target), size=(1, args.batch_factor_target_val))
    val_unlabtarget_idx = list(val_unlabtarget_idx[0])
    valnames_labtarget = [val_names_lab_target[idx] for idx in val_labtarget_idx]
    valnames_unlabtarget = [val_names_unlab_target[idx] for idx in val_unlabtarget_idx]

    print('Training files names listing...................................')
    print(trainnames_source)
    print(trainnames_labtarget)
    print(trainnames_unlabtarget)
    print(valnames_source)
    print(valnames_labtarget)
    print(valnames_unlabtarget)
    # Get source files
    brains, data, data_t1, labels, GM_distance, ventdistmap = create_data_array_from_loaded_data_ox1(
        trainnames_source, plane=mode)
    traindata_source = get_slices_from_data_with_aug(brains, data, data_t1, labels, GM_distance, ventdistmap,
                                                     plane=mode)
    brains, data, data_t1, labels, GM_distance, ventdistmap = create_data_array_from_loaded_data_ox1(
        valnames_source, plane=mode)
    valdata_source = get_slices_from_data_with_aug(brains, data, data_t1, labels, GM_distance, ventdistmap,
                                                   plane=mode, test=1)

    if dataset == 'mwsc':
        # Get target files for unlabeled data
        brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ox1_unlab(
            trainnames_unlabtarget, plane=mode)
        traindata_unlabtarget = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                                    ventdistmap, plane=mode)
        brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ox1_unlab(
            valnames_unlabtarget, plane=mode)
        valdata_unlabtarget = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                                  ventdistmap, plane=mode, test=1)

        # Get target files for labeled data
        brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ox1_unlab(
            trainnames_labtarget, plane=mode)
        traindata_labtarget = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                                  ventdistmap, plane=mode)
        brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ox1_unlab(
            valnames_labtarget, plane=mode)
        valdata_labtarget = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                                ventdistmap, plane=mode, test=1)
    else:
        # Get target files for unlabeled data
        brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ukb_unlab(
            trainnames_unlabtarget, plane=mode)
        traindata_unlabtarget = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                                    ventdistmap, plane=mode)
        brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ukb_unlab(
            valnames_unlabtarget, plane=mode)
        valdata_unlabtarget = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                                  ventdistmap, plane=mode, test=1)

        # Get target files for labeled data
        brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ukb_unlab(
            trainnames_labtarget, plane=mode)
        traindata_labtarget = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                                  ventdistmap, plane=mode)
        brains, data, data_t1, GM_distance, ventdistmap = create_data_array_from_loaded_data_ukb_unlab(
            valnames_labtarget, plane=mode)
        valdata_labtarget = get_slices_from_data_with_aug_unlab(brains, data, data_t1, GM_distance,
                                                                ventdistmap, plane=mode, test=1)

    batchgen_train_src_half = batch_generator(traindata_source, args.batch_size // 2, shuffle=True)
    batchgen_train_labtgt_half = batch_generator(traindata_labtarget, args.batch_size // 2,
                                                 shuffle=True)
    batchgen_train_unlabtgt_half = batch_generator(traindata_unlabtarget, args.batch_size // 2,
                                                   shuffle=True)
    batchgen_train_src_full = batch_generator(traindata_source, args.batch_size, shuffle=True)

    train_src_gens = [batchgen_train_src_full, batchgen_train_src_half]
    train_tgt_gens = [batchgen_train_labtgt_half, batchgen_train_unlabtgt_half]

    if ep <= args.pretrain_epochs:
        batchgen_val_src_full = batch_generator(valdata_source, args.batch_size, shuffle=False)
        val_gens = [batchgen_val_src_full]
    else:
        batchgen_val_src_full = batch_generator(valdata_source, args.batch_size, shuffle=False)
        batchgen_val_unlabtgt_half = batch_generator(valdata_unlabtarget, args.batch_size // 2, shuffle=False)
        batchgen_val_labtgt_half = batch_generator(valdata_labtarget, args.batch_size // 2, shuffle=False)
        batchgen_val_src_half = batch_generator(valdata_source, args.batch_size // 2, shuffle=False)
        val_gens = [batchgen_val_src_full, batchgen_val_src_half, batchgen_val_labtgt_half, batchgen_val_unlabtgt_half]

    return [traindata_source, valdata_source], train_src_gens, train_tgt_gens, val_gens


def numpy_to_torch_cuda(nparray, device, datatype):
    tensorcpu = torch.from_numpy(nparray)
    tensorgpu = tensorcpu.to(device=device, dtype=datatype)
    return tensorgpu