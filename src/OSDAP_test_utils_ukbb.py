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


def test_semidann_omnisup(testdata, models, device):
    [truenet, domain_predictor] = models
    truenet.eval()
    domain_predictor.eval()

    prob_array = np.array([])
    nsteps = testdata[0].shape[0]
    gen_next_test_batch = batch_generator(testdata, 1, shuffle=False)  # Batch-size = 1

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

