#!/usr/bin/env python
#   Copyright (C) 2016 University of Oxford 
#   SHBASECOPYRIGHT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import numpy as np
from torch.autograd import Variable
from model_layers_functions_torch import *

class DiceLoss(_WeightedLoss):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__(weight)

    def forward(self, pred_binary, target_binary):
        """
        Forward pass
        :param pred_binary: torch.tensor (NxCxHxW)
        :param target_binary: torch.tensor (NxHxW)
        :return: scalar
        """
        smooth = 1.
        pred_vect = pred_binary.contiguous().view(-1)
        target_vect = target_binary.contiguous().view(-1)
        intersection = (pred_vect * target_vect).sum()
        dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dice = dice.to(device=device,dtype=torch.float)
        return -dice

class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight)

    def forward(self, inputs, targets):
        """
        Forward pass
        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        targets = targets.to(device=device, dtype=torch.long)
        return self.nll_loss(inputs, targets)

class CombinedLoss(_Loss):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight=None):
        """
        Forward pass
        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        input_soft = F.softmax(input, dim=1)
        probs_vector = input_soft.contiguous().view(-1,2)
        mask_vector = (probs_vector[:,1] > 0.5).double()
        l2 = torch.mean(self.dice_loss(mask_vector, target))
        if weight is None:
            l1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            l1 = torch.mean(
                torch.mul(self.cross_entropy_loss.forward(input, target), weight.cuda()))
        return l1 + l2

    '''
class CombinedLoss(_Loss):
    def __init__(self):
        super(LossFunction, self).__init__()
        
    def dice_loss(self,pred_binary,target_binary):
        smooth = 1.
        pred_vect = pred_binary.contiguous().view(-1)
        target_vect = target_binary.contiguous().view(-1)
        intersection = (pred_vect * target_vect).sum()
        dice = (2. * intersection + smooth) /
            (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
        return -dice

    def mask_prediction2d(self, pred_logits):
        softmax_layer = nn.Softmax()
        probs = softmax_layer(pred_logits)
        probs_vector = probs.contiguous().view(-1,2)
        mask_vector = (probs_vector[:,1] > 0.5).double()
        return mask_vector
        
    def forward(self, pred, target, weights):
        criterion1 = nn.CrossEntropyLoss(reduce=False, reductio='none')
        loss_ce = criterion1(pred,target)
        weights_vector = weights.contiguous().view(-1)
        loss_ce_vector = loss_ce.contiguous().view(-1)
        loss1 = (loss_ce_vector * weights_vector) / torch.sum(weights_vector)
        binary_pred = mask_prediction2d(pred)
        loss2 = dice_loss(binary_pred,target)
        loss = loss1 + loss2
        return loss
        '''

    

