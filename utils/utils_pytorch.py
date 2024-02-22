#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:55:46 2022

@author: rouge
"""
import torch
from torch import nn
import time

import torch.nn.functional as F

from skimage.morphology import skeletonize

import sys
sys.path.append('..')
from utils.skeletonize import Skeletonize


# Utils

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
# Sigmoid
def sigmoid_clip(epsilon):
    
    def sigmoid(x):
        s = nn.Sigmoid()
        return torch.clip(s(x), epsilon, 1 - epsilon)
    
    return sigmoid


# Functions for soft-skeleton taken from https://github.com/jocpae/clDice
def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


# Apply skeletonization function from scikit-image
# Take a full tensor of size (batch_size, 1, ...) and skelotonize each volume
def skeletonize_tensor(tensor):
    """
    

    Parameters
    ----------
    tensor : Tensor of size (batch_size, 1, size_volume)
        Tensor of size (batch_size, 1, size_volume) where we want to skeletonize each volume

    Returns
    -------
    X : Tensor
        Tensor of size (batch_size, 1, size_volume) where each volume is skeletonized

    """
    X = torch.clone(tensor)
    X = X.detach().numpy()
    shape_ = X.shape
    for i in range(shape_[0]):
        x = X[i][0]
        skeleton = skeletonize(x) / 255
        X[i][0] = skeleton
    X = torch.tensor(X)
    return X


def skeletonize_numpy(tensor):
    """
    

    Parameters
    ----------
    tensor : Tensor of size (batch_size, 1, size_volume)
        Tensor of size (batch_size, 1, size_volume) where we want to skeletonize each volume

    Returns
    -------
    X : Tensor
        Tensor of size (batch_size, 1, size_volume) where each volume is skeletonized

    """
    X = tensor
    shape_ = X.shape
    for i in range(shape_[0]):
        x = X[i][0]
        skeleton = skeletonize(x) / 255
        X[i][0] = skeleton
    X = torch.tensor(X)
    return X


# %% Metrics and losses

# clDice when you already have skeletons computed
class clDice_pytorch(nn.Module):
    """"" clDice for Pytorch"""
    
    def __init__(self):
        super(clDice_pytorch, self).__init__()
        
    def forward(self, pred, pred_skel, y, y_skel):
        """
        

        Parameters
        ----------
        pred : Tensor
           Predicted segmentation - Shape (batch_size,channels,dim_image)
        pred_skel : Tensor
            Predicted centerlines - Shape (batch_size,channels,dim_image)
        y : Tensor
           Ground truth segmentations - Shape (batch_size,channels,dim_image)
        y_skel : Tensor
            Ground truth centerlines - Shape (batch_size,channels,dim_image)

        """
        epsilon = 1e-5
        dim = list(range(1, pred.dim()))
        t_prec = (torch.sum(pred_skel * y, dim=dim) + epsilon) / (torch.sum(pred_skel, dim=dim) + epsilon)
        t_sens = (torch.sum(y_skel * pred, dim=dim) + epsilon) / (torch.sum(y_skel, dim=dim) + epsilon)
        cl_dice = torch.mean(1.0 - (2. * t_prec * t_sens) / (t_prec + t_sens))
        return cl_dice
    
    
# clDice with original soft-skeleton algorithm
def cl_dice_loss(alpha, iter_):
    def cl_dice(y_pred, y_true):
        
        start = time.time()
        y_pred_skel = soft_skel(y_pred, iter_=iter_)
        y_true_skel = soft_skel(y_true, iter_=iter_)
        stop = time.time()
        print(f"Iter={iter_}")
        print(f"Time soft skeleton:{stop-start}")
        
        dice_loss = dice_loss_pytorch
        cl_dice_loss = clDice_pytorch()
        
        dice = dice_loss(y_pred, y_true)
        cl_dice = cl_dice_loss(y_pred, y_pred_skel, y_true, y_true_skel)
        loss = alpha * dice + (1 - alpha) * cl_dice
        return loss, dice, cl_dice
        
    return cl_dice


# clDice with new skeletonization algorithm
def cl_dice_loss_new_skeletonization(alpha, probabilistic=True, beta=0.33, tau=1.0, method='Boolean', num_iter=5):
    def cl_dice(y_pred, y_true):
        
        with torch.no_grad():
            skeletonization_module = Skeletonize(probabilistic=probabilistic, beta=beta, tau=tau, simple_point_detection=method, num_iter=num_iter)
        
            y_pred_skel = skeletonization_module(y_pred)
            y_true_skel = skeletonization_module(y_true)

        dice_loss = dice_loss_pytorch
        cl_dice_loss = clDice_pytorch()
        
        dice = dice_loss(y_pred, y_true)
        cl_dice = cl_dice_loss(y_pred, y_pred_skel, y_true, y_true_skel)
        loss = alpha * dice + (1 - alpha) * cl_dice
        return loss, dice, cl_dice
        
    return cl_dice
        

# Dice for Pytorch
def dice_metric_pytorch(y_pred, y_true):
    epsilon = 1e-5
    intersection = torch.sum(y_pred * y_true, dim=list(range(1, y_pred.dim())))
    union = torch.sum(y_pred, dim=list(range(1, y_pred.dim()))) + torch.sum(y_true, dim=list(range(1, y_true.dim())))
    # print('inter')
    # print(intersection)
    # print('union')
    # print(union)
    return (2. * intersection + epsilon) / (union + epsilon)


# Dice loss for Pytorch
def dice_loss_pytorch(y_pred, y_true):
    epsilon = 1e-5
    intersection = torch.sum(y_pred * y_true, dim=list(range(1, y_pred.dim())))
    union = torch.sum(y_pred, dim=list(range(1, y_pred.dim()))) + torch.sum(y_true, dim=list(range(1, y_true.dim())))
    # print('intersection')
    # print(intersection)
    # print('union')
    # print(union)
    return torch.mean(1.0 - (2. * intersection + epsilon) / (union + epsilon))
