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
from utils import *
from OSDAP_utils import append_data, dice_coeff, get_training_validation_batch_generators, \
    EarlyStoppingModelCheckpointing, numpy_to_torch_cuda
import data_preprocessing
from sklearn.utils import shuffle
import model_layers_functions_torch
from sklearn.metrics import accuracy_score
from loss_functions_torch import *
import data_preparation_pytorch_dann


def train_semidann_omnisup(args, names, dir_checkpoint, models, batch_size, num_epochs, device,
                           mode='axial', save_checkpoint=True, save_resume=True):
    [train_names_source, train_names_lab_target, train_names_unlab_target,
     val_names_source, val_names_lab_target, val_names_unlab_target] = names

    num_iters = max(len(train_names_source) // args.batch_factor_source, 1)
    print('Number of iterations: ', num_iters, flush=True)
    pretrain_epochs = args.pretrain_epochs
    [truenet, domain_predictor] = models
    lossvals = []
    losses_dom_val = []
    val_lossvals = []
    val_losses_dom_val = []
    val_dice = []
    val_dom_acc = []
    best_val_dice = 0.0

    optimizer = optim.Adam(truenet.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.2, last_epoch=-1)
    optimizer_dm = optim.SGD(list(domain_predictor.parameters()), lr=args.learning_rate_dom,
                             momentum=args.momentum)
    criterion = CombinedLoss()
    domain_criterion = nn.BCELoss()
    criteria = [criterion, domain_criterion]
    criterion.cuda()
    domain_criterion.cuda()

    gstep = 0
    start_epoch = 1
    if save_resume:
        try:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_omni_semiDANN_withdom_' + mode + '.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_omni_semiDANN_withdom_' + mode + '.pth')
            checkpoint_resumetraining = torch.load(ckpt_path)
            start_epoch = checkpoint_resumetraining['epoch'] + 1
            if start_epoch <= pretrain_epochs:
                truenet.load_state_dict(checkpoint_resumetraining['model_state_dict'])
                optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
                start_epoch = checkpoint_resumetraining['epoch'] + 1
                lossvals = checkpoint_resumetraining['loss_train']
                val_lossvals = checkpoint_resumetraining['loss_val']
                val_dice = checkpoint_resumetraining['dice_val']
                best_val_dice = checkpoint_resumetraining['best_val_dice']
            else:
                checkpoint_resumetraining = torch.load(ckpt_path)
                truenet.load_state_dict(checkpoint_resumetraining['model_state_dict'])
                domain_predictor.load_state_dict(checkpoint_resumetraining['dompred_state_dict'])
                optimizer.load_state_dict(checkpoint_resumetraining['optimizer_state_dict'])
                optimizer_dm.load_state_dict(checkpoint_resumetraining['optimizer_dom_state_dict'])
                scheduler.load_state_dict(checkpoint_resumetraining['scheduler_state_dict'])
                lossvals = checkpoint_resumetraining['loss_train']
                val_lossvals = checkpoint_resumetraining['loss_val']
                val_dice = checkpoint_resumetraining['dice_val']
                losses_dom_val = checkpoint_resumetraining['loss_dom_train']
                val_losses_dom_val = checkpoint_resumetraining['loss_dom_val']
                val_dom_acc = checkpoint_resumetraining['dom_acc']
                best_val_dice = checkpoint_resumetraining['best_val_dice']
        except:
            print('Not found any model to load and resume training!', flush=True)

    print('Training started!!.......................................')
    for epoch in range(start_epoch, num_epochs + 1):
        truenet.train()
        domain_predictor.train()
        old_model_truenet = truenet

        # For Omni supervised learning
        old_model_truenet.eval()

        print('Epoch: ' + str(epoch) + 'starting!..............................')
        for i in range(num_iters):
            trainnames = [train_names_source, train_names_lab_target, train_names_unlab_target]
            valnames = [val_names_source, val_names_lab_target, val_names_unlab_target]
            sourcedata, train_src_gens, train_tgt_gens, val_gens = \
                get_training_validation_batch_generators(args, trainnames, valnames, epoch - 1, i, mode, dataset='ukbb')

            [batchgen_train_src_full, batchgen_train_src_half] = train_src_gens
            [batchgen_train_labtgt_half, batchgen_train_unlabtgt_half] = train_tgt_gens
            [traindata_source, valdata_source] = sourcedata

            numsteps = min(traindata_source[0].shape[0] // args.batch_size, 100)
            running_loss = 0.0
            running_dom_loss = 0.0
            for j in range(numsteps):
                truenet.train()
                domain_predictor.train()

                if epoch <= pretrain_epochs:  # For pre-training
                    optimizer.zero_grad()

                    Xsf, ysf, pwgsf, pwvsf = next(batchgen_train_src_full)
                    Xsf = Xsf.transpose(0, 3, 1, 2)
                    pix_weights = (pwgsf + pwvsf).astype(float)

                    Xsf = numpy_to_torch_cuda(Xsf, device, torch.float32)
                    ysf = numpy_to_torch_cuda(ysf, device, torch.double)
                    pix_weights = numpy_to_torch_cuda(pix_weights, device, torch.float32)

                    masks_pred, bottleneck = truenet(Xsf)
                    loss = criterion(masks_pred, ysf, weight=pix_weights)
                    running_loss += loss.item()
                    del bottleneck
                    del Xsf
                    del ysf
                    del pix_weights
                    loss.backward()
                    optimizer.step()
                else:  # For semi-DANN training
                    Xsh, ysh, pwgsh, pwvsh = next(batchgen_train_src_half)
                    Xsh = Xsh.transpose(0, 3, 1, 2)
                    ysh = np.tile(ysh, (1, 1, 1, 1))
                    ysh = np.concatenate([(1 - ysh), ysh], axis=0)
                    ysh = ysh.transpose(1, 0, 2, 3)

                    Xlth, pwglth, pwvlth = next(batchgen_train_labtgt_half)
                    Xlth = Xlth.transpose(0, 3, 1, 2)
                    Xulth, _, _ = next(batchgen_train_unlabtgt_half)
                    Xulth = Xulth.transpose(0, 3, 1, 2)
                    X_dom = np.concatenate([Xsh, Xulth], axis=0)
                    dom_s = np.hstack([np.zeros([Xsh.shape[0], 1]), np.ones([Xsh.shape[0], 1])])
                    dom_t = np.hstack([np.ones([Xulth.shape[0], 1]), np.zeros([Xulth.shape[0], 1])])
                    dom_true = np.concatenate([dom_s, dom_t], axis=0)
                    X_dom, dom_true = shuffle(X_dom, dom_true, random_state=0)
                    X_dom = numpy_to_torch_cuda(X_dom, device, torch.float32)
                    dom_true = numpy_to_torch_cuda(dom_true, device, torch.float32)
                    masks_pred, bottleneck = truenet(X_dom)
                    print(X_dom.size())
                    print(bottleneck.size())
                    del masks_pred
                    dom_output = domain_predictor(bottleneck)
                    print('output predicted domain tensor size')
                    print(dom_output.size())
                    print(dom_true.size())
                    loss_dm = domain_criterion(dom_output, dom_true)
                    running_dom_loss += loss_dm.item()
                    optimizer.zero_grad()
                    optimizer_dm.zero_grad()
                    del X_dom
                    del Xulth
                    del bottleneck
                    del dom_output
					
					# Omni-supervised learning - generating labels using the old truenet model
                    Xsh = numpy_to_torch_cuda(Xsh, device, torch.float32)
                    ysh = numpy_to_torch_cuda(ysh, device, torch.double)
                    Xlth = numpy_to_torch_cuda(Xlth, device, torch.float32)
                    X_lab = torch.cat((Xsh, Xlth), 0)
                    X_lab = X_lab.to(device=device, dtype=torch.float32)
                    ylth, bottleneck = old_model_truenet(Xlth)
                    # ylth = ylth.detach().cpu().numpy()
                    # ylth = ylth[:,1,:,:]
                    del bottleneck
                    del Xsh

                    pix_weights = (pwgsh + pwvsh).astype(float)
                    pix_weights_sh = torch.from_numpy(pix_weights)
                    pix_weights = (pwglth + pwvlth).astype(float)
                    pix_weights_lth = torch.from_numpy(pix_weights)
                    # ylth = torch.from_numpy(ylth)
                    ylth = ylth.to(device=device, dtype=torch.double)
                    softmax = nn.Softmax()
                    ylth = softmax(ylth)

                    print(ysh.size())
                    print(ylth.size())
                    y_lab = torch.cat((ysh, ylth), 0)
                    pix_weights_lab = torch.cat((pix_weights_sh, pix_weights_lth), dim=0)
                    pix_weights_lab = pix_weights_lab.to(device=device, dtype=torch.float32)
                    masks_pred, bottleneck = truenet(X_lab)
                    del bottleneck
                    del Xlth
                    del X_lab
                    y_lab = torch.argmax(y_lab, dim=1)
                    y_lab = y_lab.to(device=device, dtype=torch.double)
                    print(masks_pred.dtype)
                    print(y_lab.dtype)
                    print(pix_weights_lab.dtype)
                    loss = criterion(masks_pred, y_lab, weight=pix_weights_lab)
                    del y_lab
                    del ylth
                    del ysh
                    del pix_weights_lab
                    del pix_weights_sh
                    del pix_weights_lth
                    loss_total = loss + loss_dm
                    running_loss += loss.item()
                    running_dom_loss += loss_dm.item()
                    loss_total.backward()
                    optimizer.step()
                    optimizer_dm.step()

                gstep += 1
                if j % 60 == 0:
                    val_score, domain_acc, val_loss, val_dom_loss = validate_semidann_omnisup(args, valdata_source,
                                                                                              val_gens,
                                                                                              [truenet,
                                                                                               domain_predictor],
                                                                                              [criterion,
                                                                                               domain_criterion],
                                                                                              device, epoch,
                                                                                              old_model_truenet)
                    scheduler.step(val_score)

            if i % 10 == 0:
                print('Train Mini-batch: {} out of Epoch: {} [{}/{} ({:.0f}%)]\tSeg loss: {:.6f}'.format(
                    (i + 1), epoch, (i + 1) * args.batch_factor_source, len(train_names_source),
                                    100. * (i / num_iters), loss.item()), flush=True)

            lossvals.append(running_loss / numsteps)
            print('Training set: Average loss: ', (running_loss / numsteps), flush=True)
            losses_dom_val.append(running_dom_loss / numsteps)
            val_lossvals.append(val_loss)
            print('Validation set: Loss: ', val_loss, flush=True)
            val_losses_dom_val.append(val_dom_loss)
            val_dice.append(val_score)
            val_dom_acc.append(domain_acc)

        if epoch <= args.pretrain_epochs:
            if epoch % 2 == 0:
                if save_checkpoint:
                    try:
                        os.mkdir(dir_checkpoint)
                        # logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(truenet.state_dict(),
                               dir_checkpoint + f'CP_semiDANN_omni_pretrain_epoch{epoch}_' + mode + '.pth')

        else:
            if epoch % 10 == 0:
                if save_checkpoint:
                    try:
                        os.mkdir(dir_checkpoint)
                        # logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(truenet.state_dict(), dir_checkpoint + f'CP_semiDANN_omni_epoch{epoch}_' + mode + '.pth')
                    torch.save(domain_predictor.state_dict(),
                                   dir_checkpoint + f'CP_semiDANN_omni_DomPred_epoch{epoch}_' + mode + '.pth')


        if save_resume:
            if dir_checkpoint is not None:
                ckpt_path = os.path.join(dir_checkpoint, 'tmp_model_omni_semiDANN_withdom_' + mode + '.pth')
            else:
                ckpt_path = os.path.join(os.getcwd(), 'tmp_model_omni_semiDANN_withdom_' + mode + '.pth')
            if epoch <= pretrain_epochs:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': truenet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_train': lossvals,
                    'loss_val': val_lossvals,
                    'dice_val': val_dice,
                    'best_val_dice': best_val_dice
                }, ckpt_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': truenet.state_dict(),
                    'dompred_state_dict': domain_predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_dom_state_dict': optimizer_dm.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_train': lossvals,
                    'loss_dom_train': losses_dom_val,
                    'loss_val': val_lossvals,
                    'loss_dom_val': val_losses_dom_val,
                    'dice_val': val_dice,
                    'dom_acc': val_dom_acc,
                    'best_val_dice': best_val_dice
                }, ckpt_path)

        if save_checkpoint:
            np.savez(os.path.join(dir_checkpoint, 'losses_omni_semiDANN_' + mode + '.npz'), train_seg_loss=lossvals,
                     train_dom_loss=losses_dom_val, val_seg_loss=val_lossvals, val_dom_loss=val_losses_dom_val)
            np.savez(os.path.join(dir_checkpoint, 'validation_perf_omni_semiDANN_' + mode + '.npz'), val_seg_score=val_dice,
                     val_dom_acc=val_dom_acc)

        # early_stopping(val_av_loss, val_av_dice, best_val_dice, model, epoch, optimizer, scheduler, av_loss,
        #               weights=save_weights, checkpoint=save_checkpoint, save_condition=save_case,
        #               model_path=dir_checkpoint)

        if val_score > best_val_dice:
            best_val_dice = val_score

        # if early_stopping.early_stop:
        #    print('Patience Reached - Early Stopping Activated', flush=True)
        #    if save_resume:
        #        if dir_checkpoint is not None:
        #            ckpt_path = os.path.join(dir_checkpoint, 'tmp_model.pth')
        #        else:
        #            ckpt_path = os.path.join(os.getcwd(), 'tmp_model.pth')
        #        os.remove(ckpt_path)
        #    return model
        #             sys.exit('Patience Reached - Early Stopping Activated')

        torch.cuda.empty_cache()  # Clear memory cache


def validate_semidann_omnisup(args, valdata_source, val_gens, models, criteria, device, ep, old_truenet):
    old_truenet.eval()
    pretrain_epochs = args.pretrain_epochs

    [criterion, domain_criterion] = criteria
    criterion.cuda()
    domain_criterion.cuda()

    nsteps = max(valdata_source[0].shape[0] // args.batch_size, 1)
    if ep <= pretrain_epochs:
        [batchgen_val_src_full] = val_gens
    else:
        [batchgen_val_src_full, batchgen_val_src_half, batchgen_val_labtgt_half,
         batchgen_val_unlabtgt_half] = val_gens

    dice_values = 0
    loss_val = 0
    loss_val_dom = 0
    dom_accs = 0

    for i in range(nsteps):
        [truenet, domain_predictor] = models
        truenet.eval()
        domain_predictor.eval()
        if ep <= pretrain_epochs:
            Xvs, yvs, pwgvs, pwvvs = next(batchgen_val_src_full)
            Xvs = Xvs.transpose(0, 3, 1, 2)
            pix_weightsv = (pwgvs + pwvvs).astype(float)
            Xvs = numpy_to_torch_cuda(Xvs, device, torch.float32)
            yvs = numpy_to_torch_cuda(yvs, device, torch.double)
            pix_weightsv = numpy_to_torch_cuda(pix_weightsv, device, torch.float32)
            val_pred, bottleneck = truenet(Xvs)
            del bottleneck
            val_loss = criterion(val_pred, yvs, weight=pix_weightsv)
            loss_val += val_loss.item()
            dom_acc = 0.0
        else:
            Xvs, yvs, pwgvs, pwvvs = next(batchgen_val_src_half)
            Xvs = Xvs.transpose(0, 3, 1, 2)
            Xvut, _, _ = next(batchgen_val_unlabtgt_half)
            Xvlt, pwglt, pwvlt = next(batchgen_val_labtgt_half)
            Xvut = Xvut.transpose(0, 3, 1, 2)
            Xvlt = Xvlt.transpose(0, 3, 1, 2)
            Xvd = np.concatenate([Xvs, Xvut], axis=0)
            domv_s = np.hstack([np.zeros([Xvs.shape[0], 1]), np.ones([Xvs.shape[0], 1])])
            domv_t = np.hstack([np.zeros([Xvut.shape[0], 1]), np.ones([Xvut.shape[0], 1])])
            domv_true = np.concatenate([domv_s, domv_t], axis=0)
            domv_truet = numpy_to_torch_cuda(domv_true, device, torch.float32)
            Xvd = numpy_to_torch_cuda(Xvd, device, torch.float32)
            val_pred1, bottleneck = truenet(Xvd)
            del val_pred1
            del Xvd
            dom_val_out = domain_predictor(bottleneck)
            print(dom_val_out.size())
            print(domv_truet.size())
            val_loss_dm = domain_criterion(dom_val_out, domv_truet)
            del domv_truet
            del bottleneck
            loss_val_dom += val_loss_dm.item()
            domains = np.argmax(dom_val_out.detach().cpu().numpy(), axis=1)
            true_doms = np.argmax(domv_true, axis=1)
            dom_acc = accuracy_score(true_doms, domains)

            Xvs = np.concatenate([Xvs, Xvlt], axis=0)
            Xvs = numpy_to_torch_cuda(Xvs, device, torch.float32)
            yvs = np.tile(yvs, (1, 1, 1, 1))
            yvs = np.concatenate([(1 - yvs), yvs], axis=0)
            yvs = yvs.transpose(1, 0, 2, 3)
            yvs = numpy_to_torch_cuda(yvs, device, torch.double)
            Xvlt = numpy_to_torch_cuda(Xvlt, device, torch.float32)
            yvlt, _ = old_truenet(Xvlt)
            yvlt = yvlt.to(device=device, dtype=torch.double)
            softmax = nn.Softmax()
            yvlt = softmax(yvlt)
            yvs = torch.cat((yvs, yvlt), 0)
            yvs = torch.argmax(yvs, dim=1)
            yvs = yvs.to(device=device, dtype=torch.double)
            val_pred, bottleneck = truenet(Xvs)
            del bottleneck
            del Xvlt
            del yvlt
            del Xvs
            pix_weights = (pwgvs + pwvvs).astype(float)
            pix_weightsvs = torch.from_numpy(pix_weights)
            pix_weights = (pwglt + pwglt).astype(float)
            pix_weightsvlt = torch.from_numpy(pix_weights)
            pix_weightsv = torch.cat((pix_weightsvs, pix_weightsvlt), dim=0)
            pix_weightsv = pix_weightsv.to(device=device, dtype=torch.float32)
            val_loss = criterion(val_pred, yvs, weight=pix_weightsv)
            del pix_weightsv
            loss_val += val_loss.item()

        softmax = nn.Softmax()
        probs = softmax(val_pred)
        probs_vector = probs.contiguous().view(-1, 2)
        mask_vector = (probs_vector[:, 1] > 0.5).double()
        target_vector = yvs.contiguous().view(-1)
        dice_val = dice_coeff(mask_vector, target_vector)
        del probs_vector
        del probs
        del yvs
        del mask_vector
        del target_vector

        dice_values += dice_val
        dom_accs += dom_acc
    dice_values = dice_values / (nsteps + 1)
    dom_accs = dom_accs / (nsteps + 1)
    loss_val = loss_val / (nsteps + 1)
    loss_val_dom = loss_val_dom / (nsteps + 1)
    return dice_values, dom_accs, loss_val, loss_val_dom

# def test_semidann_omnisup(testdata_source, testdata_target, models, batch_size, device, ep, mode='axial'):
#     [truenet, domain_predictor] = models
#     truenet.eval()
#     domain_predictor.eval()
#     pretrain_epochs = 51
#
#     testdata = append_data(testdata)
#     prob_array = np.array([])
#     if ep <= pretrain_epochs:
#         nsteps = max(testdata_source[0].shape[0] // batch_size, 1)
#         gen_next_test_batch_source_full = batch_generator(testdata_source, batch_size, shuffle=False)
#     else:
#         gen_next_test_batch_source_full = batch_generator(testdata_source, batch_size, shuffle=False)
#         nsteps = max(testdata_source[0].shape[0] // (batch_size // 2), testdata_target[0].shape[0] // (batch_size // 2))
#         gen_next_test_batch_target_half = batch_generator(testdata_target, batch_size // 2, shuffle=False)
#         gen_next_test_batch_source_half = batch_generator(testdata_source, batch_size // 2, shuffle=False)
#     dice_values = 0
#     for i in range(nsteps):
#         Xv, yv, pwgv, pwvv = next(gen_next_test_batch_source_full)
#         Xv = Xv.transpose(0, 3, 1, 2)
#         if ep > pretrain_epochs:
#             Xvs, _, _, _ = next(gen_next_test_batch_source_half)
#             Xvs = Xvs.transpose(0, 3, 1, 2)
#             Xvt, _, _, _ = next(gen_next_test_batch_target_half)
#             Xvt = Xvt.transpose(0, 3, 1, 2)
#             Xvd = np.concatenate([Xvs, Xvt], axis=0)
#             domv_s = np.hstack([np.zeros([Xvs.shape[0], 1]), np.ones([Xvs.shape[0], 1])])
#             domv_t = np.hstack([np.zeros([Xvt.shape[0], 1]), np.ones([Xvt.shape[0], 1])])
#             domv_true = np.concatenate([domv_s, domv_t], axis=0)
#         print('Testing/validation dimensions.......................................')
#         print(Xv.shape)
#         print(yv.shape)
#         pix_weights_gmv = pwgv
#         pix_weights_ventv = pwvv
#         pix_weightsv = (pix_weights_gmv + pix_weights_ventv) * (pix_weights_ventv > 0).astype(float)
#         Xv = torch.from_numpy(Xv)
#         Xv = Xv.to(device=device, dtype=torch.float32)
#         val_pred, bottleneck = truenet(Xv)
#         if ep <= pretrain_epochs:
#             del bottleneck
#             dom_acc = 90
#         else:
#             Xvd = torch.from_numpy(Xvd)
#             Xvd = Xvd.to(device=device, dtype=torch.float32)
#             val_pred1, bottleneck = truenet(Xvd)
#             del val_pred1
#             dom_val_out = domain_predictor(bottleneck)
#             domains = np.argmax(dom_val_out.detach().cpu().numpy(), axis=1)
#             true_doms = np.argmax(domv_true, axis=1)
#             dom_acc = accuracy_score(true_doms, domains)
#         softmax = nn.Softmax()
#         probs = softmax(val_pred)
#         probs_vector = probs.contiguous().view(-1, 2)
#         mask_vector = (probs_vector[:, 1] > 0.5).double()
#         yv = torch.from_numpy(yv)
#         yv = yv.to(device=device, dtype=torch.double)
#         target_vector = yv.contiguous().view(-1)
#         dice_val = dice_coeff(mask_vector, target_vector)
#         probs1 = probs.cpu()
#         probs_np = probs1.detach().numpy()
#
#         prob_array = np.concatenate((prob_array, probs_np), axis=0) if prob_array.size else probs_np
#
#         dice_values += dice_val
#     prob_array = prob_array.transpose(0, 2, 3, 1)
#     dice_values = dice_values / (nsteps + 1)
#     return dice_values, dom_acc, prob_array

