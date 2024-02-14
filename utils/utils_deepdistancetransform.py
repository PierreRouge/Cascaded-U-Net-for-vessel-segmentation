#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:41:38 2024

@author: rouge
"""
import numpy as np
import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import nibabel as nib

from glob import glob

from scipy.ndimage.morphology import distance_transform_edt

import sys
sys.path.append('..')
from utils.utils_pytorch import dice_metric_pytorch, dice_loss_pytorch, sigmoid_clip


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

sigmoid = sigmoid_clip(1E-3)
softmax = nn.Softmax(dim=1)


def distance_transform(binary_array):
    dtm = np.round(distance_transform_edt(binary_array))
    return dtm
    

# Training
def train_loop_DeepDistance(dataloader, validloader, model, loss_param, input_, optimizer, device, epoch, max_epoch):

    if loss_param == "deep-distance-transform":
        loss_0 = dice_loss_pytorch
        loss_1 = nn.BCELoss()
        loss_ce_dtm = nn.CrossEntropyLoss()
        
        model.eval()
        val_loss_0 = 0.0
        val_dice_0 = 0.0
        with torch.no_grad():
            for batch, data in enumerate(validloader):
                if input_ == 'MRI':
                    X_val = data['image']
                elif input_ == 'segmentation':
                    X_val = data['segmentation']
                elif input_ == 'Both':
                    X_val_1 = data['image']
                    X_val_2 = data['segmentation']
                    X_val = torch.cat((X_val_1, X_val_2), dim=1)
                
                y_val_seg = data['GT']
                y_val_dtm = data['DTM']
                y_val_seg = y_val_seg.to(device)
                y_val_dtm = y_val_dtm.to(device)
                b, c, s1, s2, s3 = y_val_dtm.shape
                
                X_val = X_val.float()
                X_val = X_val.to(device)
                
                pred_val_seg, pred_val_dtm = model(X_val)
                pred_val_seg = sigmoid(pred_val_seg)
                # pred_val_dtm = softmax(pred_val_dtm)
    
                val_loss = loss_0(pred_val_seg, y_val_seg) + loss_1(pred_val_seg, y_val_seg) + loss_ce_dtm(pred_val_dtm, y_val_dtm.view(b, s1, s2, s3).long())
                val_loss = val_loss.item()
                val_loss_0 += val_loss
                
                pred_val = nn.functional.threshold(pred_val_seg, threshold=0.5, value=0)
                ones = torch.ones(pred_val.shape, dtype=torch.float)
                ones = ones.to(device)
                pred_val_seg = torch.where(pred_val_seg > 0, ones, pred_val)
                dice_val = dice_metric_pytorch(pred_val_seg, y_val_seg)
                dice_val = dice_val.cpu().detach().numpy()
                dice_val = np.mean(dice_val)
                val_dice_0 += dice_val.item()
    
            val_loss = val_loss_0 / len(validloader)
            val_dice = val_dice_0 / len(validloader)
            
        train_loss = 0.0
        start = time.time()
        for batch_train, data in enumerate(dataloader):
            if input_ == 'MRI':
                X = data['image']
            elif input_ == 'segmentation':
                X = data['segmentation']
            elif input_ == 'Both':
                X_1 = data['image']
                X_2 = data['segmentation']
                X = torch.cat((X_1, X_2), dim=1)
            
            y_seg = data['GT']
            y_dtm = data['DTM']
            
            y_seg = y_seg.to(device)
            y_dtm = y_dtm.to(device)
            
            b, c, s1, s2, s3 = y_dtm.shape
            
            # Compute prediction and loss
            X = X.float()
            X = X.to(device)
            pred_seg, pred_dtm = model(X)
            pred_seg = sigmoid(pred_seg)

            loss_dice = loss_0(pred_seg, y_seg)
            loss_bce = loss_1(pred_seg, y_seg)  
            loss_dtm = loss_ce_dtm(pred_dtm, y_dtm.view(b, s1, s2, s3).long())
            loss = loss_dice + loss_bce + loss_dtm
            train_loss += loss.item()

            pred_seg = nn.functional.threshold(pred_seg, threshold=0.5, value=0)
            ones = torch.ones(pred_seg.shape, dtype=torch.float)
            ones = ones.to(device)
            pred = torch.where(pred_seg > 0, ones, pred_seg)
            dice = dice_metric_pytorch(pred_seg, y_seg)
            dice = dice.cpu().detach().numpy()
            dice = np.mean(dice)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader)
        end = time.time()
        epoch_duration = end - start
       
        logs = {"train_loss": train_loss,
                "loss_dice": loss_dice.item(),
                "loss_bce": loss_bce.item(),
                "loss_dtm": loss_dtm.item(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "epoch_duration": epoch_duration}
        
        print(f"loss: {train_loss:>7f}, val_loss:{val_loss:>7f}, dice_val:{val_dice:>7f}, dice:{dice:>7f}, time:{epoch_duration:>7f}")
        
        return logs
   
    
if __name__ == '__main__':
    dir_data = '../../../Thèse_Rougé_Pierre/Data/Bullit/raw/GT/*'
    
    for f in glob(dir_data):
        image = nib.load(f)
        data = image.get_fdata()
            
        dtm = distance_transform(data)
        
        dtm_nii = nib.Nifti1Image(dtm, affine=image.affine, header=image.header)
        
        nib.save(dtm_nii, f.replace('GT', 'DistanceMap'))