#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:16:03 2022

@author: rouge
"""

import sys
from torch import nn

sys.path.append('..')
from utils.utils_pytorch import dice_loss_pytorch, clDice_pytorch


# Standard losses for multitask segmentation

class loss_dice_multitask(nn.Module):
    """"" Loss dice for multitask learning pytorch
    
    Segmentation : Dice
    Skeletonization: Dice"""
    
    def __init__(self, alpha):
        """
        Init class object

        Parameters
        ----------
        alpha : float
            Float bewteen 0 and 1 to tune the imortance of each loss task

        Returns
        -------
        None.

        """
        super(loss_dice_multitask, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, pred_skel, y, y_skel):
        """
        Compute the loss

        Parameters
        ----------
        pred : Torch tensor
            Predicted segmentation size=(batch_size,1,size_volume)
        pred_skel : Torch tensor
            Predicted skeleton segmentation size=(batch_size,1,size_volume)
        y : Torch tensor
            Ground truth segmentation size=(batch_size,1,size_volume)
        y_skel : Torch tensor
            Ground truth skeleton segmentation size=(batch_size,1,size_volume)

        Returns
        -------
        loss : Torch Tensor (1,)
            Sum of losses
        loss_Dice_1 : Torch Tensor (1,)
            Loss of first task
        loss_Dice_2 : Torch Tensor (1,)
            Loss of seconde task

        """

        loss = dice_loss_pytorch
        loss_Dice_1 = loss(pred, y)
        loss_Dice_2 = loss(pred_skel, y_skel)
        loss = self.alpha * loss_Dice_1 + (1 - self.alpha) * loss_Dice_2
        return loss, loss_Dice_1, loss_Dice_2


class loss_CE_multitask(nn.Module):
    """"" Cross-entropy loss for multitask learning pytorch
    Segmentation : CE
    Skeletonization: CE"""
    
    def __init__(self, alpha):
        """
        Init class object

        Parameters
        ----------
        alpha : Float
            Float between 0 and 1 to tune the imortance of each loss task

        Returns
        -------
        None.

        """
        super(loss_CE_multitask, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, pred_skel, y, y_skel):
        """
        Compute the loss

        Parameters
        ----------
        pred : Torch tensor
            Predicted segmentation size=(batch_size,1,size_volume)
        pred_skel : Torch tensor
            Predicted skeleton segmentation size=(batch_size,1,size_volume)
        y : Torch tensor
            Ground truth segmentation size=(batch_size,1,size_volume)
        y_skel : Torch tensor
            Ground truth skeleton segmentation size=(batch_size,1,size_volume)

        Returns
        -------
        ------
        loss : Torch Tensor (1,)
            Sum of losses
        loss_Dice_1 : Torch Tensor (1,)
            Loss of first task
        loss_Dice_2 : Torch Tensor (1,)
            Loss of second task

        """
        loss_1 = nn.BCELoss()

        loss_CE_1 = loss_1(pred, y)
        loss_CE_2 = loss_1(pred_skel, y_skel)
        loss = self.alpha * loss_CE_1 + (1 - self.alpha) * loss_CE_2
        return loss, loss_CE_1, loss_CE_2
    
    
class loss_dice_CE_multitask(nn.Module):
    """"" Addition of cross-entropy loss and dice loss for multitask learning pytorch
    Segmentation : Dice + CE
    Skeletonization: Dice + CE
    """
    
    def __init__(self, alpha, alpha_dice_ce):
        """
        Init class object

        Parameters
        ----------
        alpha : Float
            Float between 0 and 1 to tune the imortance of each loss task

        Returns
        -------
        None.

        """
        super(loss_dice_CE_multitask, self).__init__()
        self.alpha = alpha
        self.alpha_dice_ce = alpha_dice_ce
    
    def forward(self, pred, pred_skel, y, y_skel):
        """
        Compute the loss

        Parameters
        ----------
        pred : Torch tensor
            Predicted segmentation size=(batch_size,1,size_volume)
        pred_skel : Torch tensor
            Predicted skeleton segmentation size=(batch_size,1,size_volume)
        y : Torch tensor
            Ground truth segmentation size=(batch_size,1,size_volume)
        y_skel : Torch tensor
            Ground truth skeleton segmentation size=(batch_size,1,size_volume)

        Returns
        -------
        ------
        loss : Torch Tensor (1,)
            Sum of losses
        loss_Dice_1 : Torch Tensor (1,)
            Loss of first task
        loss_Dice_2 : Torch Tensor (1,)
            Loss of second task

        """
        loss_1 = dice_loss_pytorch
        loss_2 = nn.BCELoss()

        dice_segmentation = loss_1(pred, y)
        CE_segmentation = loss_2(pred, y)
        dice_skeleton = loss_1(pred_skel, y_skel)
        CE_skeleton = loss_2(pred_skel, y_skel)
        loss_combine_1 = self.alpha_dice_ce * dice_segmentation + (1 - self.alpha_dice_ce) * CE_segmentation
        loss_combine_2 = self.alpha_dice_ce * dice_skeleton + (1 - self.alpha_dice_ce) * CE_skeleton
        loss = self.alpha * loss_combine_1 + (1 - self.alpha) * loss_combine_2
        
        return loss, loss_combine_1, loss_combine_2, dice_segmentation, CE_segmentation, dice_skeleton, CE_skeleton


# Loss combining Dice and clDice for multitask learning

class loss_dice_cldice_multitask(nn.Module):
    
    """ Custom loss for multitask training :
       Segmentation : Dice
       Skeletonization: Dice
       Both : clDice"""
        
    def __init__(self, lambda1, lambda2, lambda3):
        """
        

        Parameters
        ----------
        lambda1 : float
            Control weight of segmentation loss
        lambda2 : float
            Control weight of clDice loss
        alpha : float
            Parameter for focal loss to balance foreground and background
        gamma : int
            Parameter for focal loss to balance missclassified and well classified sample
        """
        
        super(loss_dice_cldice_multitask, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
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
        
        dice_loss = dice_loss_pytorch
        cl_dice_loss = clDice_pytorch()
        
        loss_1 = self.lambda1 * dice_loss(pred, y)
        loss_2 = self.lambda2 * dice_loss(pred_skel, y_skel)
        loss_3 = self.lambda3 * cl_dice_loss(pred, pred_skel, y, y_skel)
        
        loss = loss_1 + loss_2 + loss_3
        
        return loss, loss_1, loss_2, loss_3
